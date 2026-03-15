"""Darknet network definition and YOLO helper functions.

Created: Aug 2024
Purpose: Provides model parsing/loading/inference helpers used by recognition scripts.
"""

import numpy as np
import torch
import torch.nn as nn


class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=None, num_classes=0, anchors=None, num_anchors=1):
        super().__init__()
        self.anchor_mask = anchor_mask or []
        self.num_classes = num_classes
        self.anchors = anchors or []
        self.num_anchors = max(1, int(num_anchors))
        self.anchor_step = len(self.anchors) // self.num_anchors if self.anchors else 0
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = 32
        self.seen = 0

    def forward(self, output, nms_thresh):
        self.thresh = nms_thresh
        masked_anchors = []

        anchor_step = int(self.anchor_step)
        for m in self.anchor_mask:
            start = m * anchor_step
            end = (m + 1) * anchor_step
            masked_anchors += self.anchors[start:end]

        masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        boxes = get_region_boxes(
            output.data,
            self.thresh,
            self.num_classes,
            masked_anchors,
            len(self.anchor_mask),
        )
        return boxes


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        batch_size = x.data.size(0)
        channels = x.data.size(1)
        height = x.data.size(2)
        width = x.data.size(3)

        return (
            x.view(batch_size, channels, height, 1, width, 1)
            .expand(batch_size, channels, height, stride, width, stride)
            .contiguous()
            .view(batch_size, channels, height * stride, width * stride)
        )


class EmptyModule(nn.Module):
    def forward(self, x):
        return x


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks)
        self.loss = self.models[len(self.models) - 1]

        self.width = int(self.blocks[0]["width"])
        self.height = int(self.blocks[0]["height"])

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, nms_thresh):
        index = -2
        self.loss = None
        outputs = {}
        out_boxes = []

        for block in self.blocks:
            index += 1
            block_type = block["type"]
            if block_type == "net":
                continue
            if block_type in ["convolutional", "upsample"]:
                x = self.models[index](x)
                outputs[index] = x
            elif block_type == "route":
                layers = block["layers"].split(",")
                layers = [int(i) if int(i) > 0 else int(i) + index for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                outputs[index] = x
            elif block_type == "shortcut":
                from_layer = int(block["from"])
                from_layer = from_layer if from_layer > 0 else from_layer + index
                x = outputs[from_layer] + outputs[index - 1]
                outputs[index] = x
            elif block_type == "yolo":
                boxes = self.models[index](x, nms_thresh)
                out_boxes.append(boxes)
            else:
                print(f"unknown type {block_type}")

        return out_boxes

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()

        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0

        for block in blocks:
            block_type = block["type"]
            if block_type == "net":
                prev_filters = int(block["channels"])
                continue

            if block_type == "convolutional":
                conv_id += 1
                batch_normalize = int(block["batch_normalize"])
                filters = int(block["filters"])
                kernel_size = int(block["size"])
                stride = int(block["stride"])
                is_pad = int(block["pad"])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block["activation"]

                model = nn.Sequential()
                if batch_normalize:
                    model.add_module(
                        f"conv{conv_id}",
                        nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False),
                    )
                    model.add_module(f"bn{conv_id}", nn.BatchNorm2d(filters))
                else:
                    model.add_module(
                        f"conv{conv_id}", nn.Conv2d(prev_filters, filters, kernel_size, stride, pad)
                    )

                if activation == "leaky":
                    model.add_module(f"leaky{conv_id}", nn.LeakyReLU(0.1, inplace=True))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)

            elif block_type == "upsample":
                stride = int(block["stride"])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                models.append(Upsample(stride))

            elif block_type == "route":
                layers = block["layers"].split(",")
                index = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + index for i in layers]

                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert layers[0] == index - 1
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())

            elif block_type == "shortcut":
                index = len(models)
                prev_filters = out_filters[index - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[index - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())

            elif block_type == "yolo":
                yolo_layer = YoloLayer()
                anchors = block["anchors"].split(",")
                anchor_mask = block["mask"].split(",")
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block["classes"])
                yolo_layer.num_anchors = int(block["num"])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)

            else:
                print(f"unknown type {block_type}")

        return models

    def load_weights(self, weightfile):
        with open(weightfile, "rb") as fp:
            header = np.fromfile(fp, count=5, dtype=np.int32)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            buf = np.fromfile(fp, dtype=np.float32)

        start = 0
        index = -2
        counter = 3

        for block in self.blocks:
            if start >= buf.size:
                break

            index += 1
            block_type = block["type"]

            if block_type == "net":
                continue
            if block_type == "convolutional":
                model = self.models[index]
                batch_normalize = int(block["batch_normalize"])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])

            percent_comp = (counter / len(self.blocks)) * 100
            print(
                f"Loading weights. Please Wait...{percent_comp:.2f}% Complete",
                end="\r",
                flush=True,
            )
            counter += 1


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes(
    output,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    only_objectness=1,
    validation=False,
):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)

    batch = output.size(0)
    assert output.size(1) == (5 + num_classes) * num_anchors
    height = output.size(2)
    width = output.size(3)

    all_boxes = []
    output = (
        output.view(batch * num_anchors, 5 + num_classes, height * width)
        .transpose(0, 1)
        .contiguous()
        .view(5 + num_classes, batch * num_anchors * height * width)
    )

    grid_x = (
        torch.linspace(0, width - 1, width)
        .repeat(height, 1)
        .repeat(batch * num_anchors, 1, 1)
        .view(batch * num_anchors * height * width)
        .type_as(output)
    )
    grid_y = (
        torch.linspace(0, height - 1, height)
        .repeat(width, 1)
        .t()
        .repeat(batch * num_anchors, 1, 1)
        .view(batch * num_anchors * height * width)
        .type_as(output)
    )

    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, height * width).view(batch * num_anchors * height * width).type_as(output)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, height * width).view(batch * num_anchors * height * width).type_as(output)

    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax(dim=1)(output[5 : 5 + num_classes].transpose(0, 1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    sz_hw = height * width
    sz_hwa = sz_hw * num_anchors

    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)

    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))

    for b in range(batch):
        boxes = []
        for cy in range(height):
            for cx in range(width):
                for i in range(num_anchors):
                    index = b * sz_hwa + i * sz_hw + cy * width + cx
                    if only_objectness:
                        conf = det_confs[index]
                    else:
                        conf = det_confs[index] * cls_max_confs[index]

                    if conf > conf_thresh:
                        box = [
                            xs[index] / width,
                            ys[index] / height,
                            ws[index] / width,
                            hs[index] / height,
                            det_confs[index],
                            cls_max_confs[index],
                            cls_max_ids[index],
                        ]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[index][c]
                                if c != cls_max_ids[index] and det_confs[index] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)

    return all_boxes


def parse_cfg(cfgfile):
    blocks = []
    block = None

    with open(cfgfile, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("["):
                if block:
                    blocks.append(block)
                block = {"type": line.lstrip("[").rstrip("]")}
                if block["type"] == "convolutional":
                    block["batch_normalize"] = "0"
            else:
                if block is None:
                    continue
                current_block = block
                key, value = line.split("=", 1)
                key = key.strip()
                if key == "type":
                    key = "_type"
                current_block[key] = value.strip()

    if block:
        blocks.append(block)

    return blocks


def print_cfg(blocks):
    print("layer     filters    size              input                output")
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    index = -2

    for block in blocks:
        index += 1
        block_type = block["type"]

        if block_type == "net":
            prev_width = int(block["width"])
            prev_height = int(block["height"])
            continue

        if block_type == "convolutional":
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            is_pad = int(block["pad"])
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            print(
                "%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    index,
                    "conv",
                    filters,
                    kernel_size,
                    kernel_size,
                    stride,
                    prev_width,
                    prev_height,
                    prev_filters,
                    width,
                    height,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block_type == "upsample":
            stride = int(block["stride"])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print(
                "%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    index,
                    "upsample",
                    stride,
                    prev_width,
                    prev_height,
                    prev_filters,
                    width,
                    height,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block_type == "route":
            layers = block["layers"].split(",")
            layers = [int(i) if int(i) > 0 else int(i) + index for i in layers]
            if len(layers) == 1:
                print("%5d %-6s %d" % (index, "route", layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print("%5d %-6s %d %d" % (index, "route", layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert prev_width == out_widths[layers[1]]
                assert prev_height == out_heights[layers[1]]
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]

            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block_type in ["region", "yolo"]:
            print("%5d %-6s" % (index, "detection"))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block_type == "shortcut":
            from_id = int(block["from"])
            from_id = from_id if from_id > 0 else from_id + index
            print("%5d %-6s %d" % (index, "shortcut", from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        else:
            print(f"unknown type {block_type}")


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start += num_b
    conv_model.weight.data.copy_(
        torch.from_numpy(buf[start : start + num_w]).view_as(conv_model.weight.data)
    )
    start += num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()

    bn_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start += num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start += num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start : start + num_b]))
    start += num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start : start + num_b]))
    start += num_b

    conv_model.weight.data.copy_(
        torch.from_numpy(buf[start : start + num_w]).view_as(conv_model.weight.data)
    )
    start += num_w
    return start
