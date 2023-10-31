# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        depth_module: dict = None,
        dense_depth_module: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "category_ids",
        gt_reg_key: str = "boxes",
        reg_weights=None,
        operation_order: Optional[List[str]] = None,
        kps_generator=None,
        max_queue_length=0,
        cls_threshold_to_reg=-1,
        init_cfg=None,
        embed_dims=None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.max_queue_length = max_queue_length
        self.cls_threshold_to_reg = cls_threshold_to_reg
        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.depth_module = build(depth_module, PLUGIN_LAYERS)
        self.dense_depth_module = build(dense_depth_module, PLUGIN_LAYERS)
        self.kps_generator = build(kps_generator, PLUGIN_LAYERS)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        feature_queue=None,
        meta_queue=None,
        training=False,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(batch_size, metas, training)
        anchor_embed = self.anchor_encoder(anchor)
        # instance_feature = instance_feature.view(batch_size, -1, self.embed_dims)
        # anchor_embed = anchor_embed.view(batch_size, -1, self.embed_dims)
        # import ipdb; ipdb.set_trace()
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        _feature_queue = self.instance_bank.feature_queue
        _meta_queue = self.instance_bank.meta_queue
        if feature_queue is not None and _feature_queue is not None:
            feature_queue = feature_queue + _feature_queue
            meta_queue = meta_queue + _meta_queue
        elif feature_queue is None:
            feature_queue = _feature_queue
            meta_queue = _meta_queue

        prediction = []
        classification = []
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                    feature_queue=feature_queue,
                    meta_queue=meta_queue,
                    depth_module=self.depth_module,
                    anchor_encoder=self.anchor_encoder,
                )
            elif op == "refine":
                anchor, cls = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        return classification, prediction

    @force_fp32(apply_to=("cls_scores", "reg_preds"))
    def loss(self, cls_scores, reg_preds, data, feature_maps=None):
        output = {}
        # import ipdb; ipdb.set_trace()
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            reg = reg[..., : len(self.reg_weights)]
            reg_front = reg[:, :900, :]
            cls_front = cls[:, :900, :]
            reg_back = reg[:, 900:, :]
            cls_back = cls[:, 900:, :]
            cls_target_front, reg_target_front, reg_weights_front = self.sampler.sample(
                cls_front,
                reg_front,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            cls_target_back, reg_target_back, reg_weights_back = self.sampler.sample(
                cls_back,
                reg_back,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )

            reg_target_front = reg_target_front[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target_front == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg_front.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls_front.max(dim=-1).values.sigmoid() > threshold
                )

            cls_front = cls_front.flatten(end_dim=1)
            cls_target_front = cls_target_front.flatten(end_dim=1)
            cls_loss_front = self.loss_cls(cls_front, cls_target_front, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights_front = reg_weights_front * reg_front.new_tensor(self.reg_weights)
            reg_target_front = reg_target_front.flatten(end_dim=1)[mask]
            reg_front = reg_front.flatten(end_dim=1)[mask]
            reg_weights_front = reg_weights_front.flatten(end_dim=1)[mask]
            reg_target_front = torch.where(
                reg_target_front.isnan(), reg_front.new_tensor(0.0), reg_target_front
            )
            reg_loss_front = self.loss_reg(
                reg_front, reg_target_front, weight=reg_weights_front, avg_factor=num_pos
            )


            reg_target_back = reg_target_back[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target_back == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg_back.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls_back.max(dim=-1).values.sigmoid() > threshold
                )

            cls_back = cls_back.flatten(end_dim=1)
            cls_target_back = cls_target_back.flatten(end_dim=1)
            cls_loss_back = self.loss_cls(cls_back, cls_target_back, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights_back = reg_weights_back * reg_back.new_tensor(self.reg_weights)
            reg_target_back = reg_target_back.flatten(end_dim=1)[mask]
            reg_back = reg_back.flatten(end_dim=1)[mask]
            reg_weights_back = reg_weights_back.flatten(end_dim=1)[mask]
            reg_target_back = torch.where(
                reg_target_back.isnan(), reg_back.new_tensor(0.0), reg_target_back
            )
            reg_loss_back = self.loss_reg(
                reg_back, reg_target_back, weight=reg_weights_back, avg_factor=num_pos
            )

            output.update(
                {
                    f"loss_cls_{decoder_idx}_front": cls_loss_front,
                    f"loss_reg_{decoder_idx}_front": reg_loss_front,
                    f"loss_cls_{decoder_idx}_back": cls_loss_back,
                    f"loss_reg_{decoder_idx}_back": reg_loss_back,
                }
            )

        if (
            self.depth_module is not None
            and self.kps_generator is not None
            and feature_maps is not None
        ):
            reg_target = self.sampler.encode_reg_target(
                data[self.gt_reg_key], reg_preds[0].device
            )
            loss_depth = []
            for i in range(len(reg_target)):
                if len(reg_target[i]) == 0:
                    continue
                key_points = self.kps_generator(reg_target[i][None])
                features = (
                    DFG.feature_sampling(
                        [f[i : i + 1] for f in feature_maps],
                        key_points,
                        data["projection_mat"][i : i + 1],
                        data["image_wh"][i : i + 1],
                    )
                    .mean(2)
                    .mean(2)
                )
                depth_confidence = self.depth_module(
                    features, reg_target[i][None, :, None], output_conf=True
                )
                loss_depth.append(-torch.log(depth_confidence).sum())
            output["loss_depth"] = (
                sum(loss_depth) / num_pos / self.kps_generator.num_pts
            )

        if self.dense_depth_module is not None:
            output["loss_dense_depth"] = self.dense_depth_module(
                feature_maps,
                focal=data.get("focal"),
                gt_depths=data["gt_depth"],
            )
        return output

    @force_fp32(apply_to=("cls_scores", "reg_preds"))
    def post_process(self, cls_scores, reg_preds):
        return self.decoder.decode(cls_scores, reg_preds)
