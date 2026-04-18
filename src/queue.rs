use bevy::{
    core_pipeline::prepass::ViewPrepassTextures,
    pbr::{ExtractedAtmosphere, MeshPipelineKey, RenderMeshInstances},
    prelude::*,
};
use bevy::render::{
    mesh::{RenderMesh, allocator::MeshAllocator},
    render_asset::RenderAssets,
    render_phase::{BinnedRenderPhaseType, DrawFunctions, ViewBinnedRenderPhases},
    render_resource::{PipelineCache, SpecializedMeshPipelines},
    view::{ExtractedView, RenderVisibleEntities},
};

use crate::{
    DrawOutline,
    mask::{OutlineBatchSetKey, OutlineBinKey},
};

use super::{ExtractedOutline, MeshOutline3d, OutlineCamera, mask_pipeline::MeshMaskPipeline};

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn queue_outline(
    outlined_meshes: Query<&ExtractedOutline>,
    draw_functions: Res<DrawFunctions<MeshOutline3d>>,
    mut outline_phases: ResMut<ViewBinnedRenderPhases<MeshOutline3d>>,
    mesh_outline_pipeline: Res<MeshMaskPipeline>,
    mut mesh_outline_pipelines: ResMut<SpecializedMeshPipelines<MeshMaskPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mesh_allocator: Res<MeshAllocator>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    views: Query<
        (
            Entity,
            &ExtractedView,
            &RenderVisibleEntities,
            &Msaa,
            Option<&ViewPrepassTextures>,
            Has<ExtractedAtmosphere>,
        ),
        With<OutlineCamera>,
    >,
) {
    let draw_function = draw_functions.read().id::<DrawOutline>();

    for (_view_entity, view, visible_entities, msaa, prepass_textures, has_atmosphere) in views.iter() {
        let Some(outline_phase) = outline_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        // Build view key from prepass textures (handles depth, normal, motion, deferred)
        if let Some(prepass) = prepass_textures {
            if prepass.depth.is_some() {
                view_key |= MeshPipelineKey::DEPTH_PREPASS;
            }
            if prepass.normal.is_some() {
                view_key |= MeshPipelineKey::NORMAL_PREPASS;
            }
            if prepass.motion_vectors.is_some() {
                view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
            }
            if prepass.deferred.is_some() {
                view_key |= MeshPipelineKey::DEFERRED_PREPASS;
            }
        }

        if has_atmosphere {
            view_key |= MeshPipelineKey::ATMOSPHERE;
        }

        let Some(render_visible_mesh_entities) = visible_entities.get::<Mesh3d>() else {
            continue;
        };

        for &(render_entity, main_entity) in &render_visible_mesh_entities.entities_cpu_culling {
            if outlined_meshes.get(render_entity).is_err() {
                continue;
            }
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(main_entity)
            else {
                tracing::warn!(target: "bevy_mesh_outline", "No mesh instance found for entity {:?}", main_entity);
                continue;
            };

            let Some(mesh_slabs) = mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id()) else {
                continue;
            };

            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id()) else {
                tracing::warn!(target: "bevy_mesh_outline", "No mesh found for entity {:?}", main_entity);
                continue;
            };

            let mut mesh_key = view_key;
            mesh_key |= MeshPipelineKey::from_primitive_topology_and_strip_index(mesh.primitive_topology(), mesh.index_format())
                | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits());

            let Ok(pipeline_id) = mesh_outline_pipelines.specialize(
                &pipeline_cache,
                &mesh_outline_pipeline,
                mesh_key,
                &mesh.layout,
            ) else {
                tracing::warn!(target: "bevy_mesh_outline", "Failed to specialize mesh pipeline");
                continue;
            };

            outline_phase.add(
                OutlineBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function,
                    vertex_slab: mesh_slabs.vertex_slab_id,
                    index_slab: mesh_slabs.index_slab_id,
                },
                OutlineBinKey {
                    asset_id: mesh_instance.mesh_asset_id().untyped(),
                },
                (render_entity, main_entity),
                mesh_instance.current_uniform_index,
                BinnedRenderPhaseType::UnbatchableMesh,
            );
        }
    }
}
