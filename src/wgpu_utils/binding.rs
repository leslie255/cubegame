use std::any::type_name;

pub trait Bindable {
    fn bind_group_layout_entry(
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayoutEntry;
    fn bind_group_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_>;
}

impl<T: Bindable> Bindable for &T {
    fn bind_group_layout_entry(
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayoutEntry {
        T::bind_group_layout_entry(binding, visibility)
    }

    fn bind_group_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        <T as Bindable>::bind_group_entry(self, binding)
    }
}

pub trait AsBindGroup {
    fn bind_group_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry>;
    fn bind_group_entries(&self) -> Vec<wgpu::BindGroupEntry<'_>>;
}

/// TODO: Make this into a derive macro so it supports structs with generic parameters.
#[macro_export]
macro_rules! impl_as_bind_group {
    ($T:path { $($binding_id:literal => $field:ident : $type:ty),* $(,)? } $($tts:tt)*) => {
        impl $crate::wgpu_utils::AsBindGroup for $T {
            fn bind_group_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
                ::std::vec![
                    $(<$type as $crate::wgpu_utils::Bindable>::bind_group_layout_entry(
                        $binding_id,
                        wgpu::ShaderStages::all(),
                    )),*
                ]
            }
            fn bind_group_entries(&self) -> Vec<wgpu::BindGroupEntry<'_>> {
                ::std::vec![
                    $($crate::wgpu_utils::Bindable::bind_group_entry(
                        &self.$field,
                        $binding_id,
                    )),*
                ]
            }
        }
        $crate::impl_as_bind_group! { $($tts)* }
    };
    () => {}
}

pub fn create_bind_group_layout<BG: AsBindGroup>(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let label = Some(type_name::<BG>());
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label,
        entries: &BG::bind_group_layout_entries(),
    })
}

pub fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bind_group: &impl AsBindGroup,
) -> wgpu::BindGroup {
    let label = Some(std::any::type_name_of_val(&bind_group));
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label,
        layout,
        entries: &bind_group.bind_group_entries(),
    })
}

impl Bindable for wgpu::TextureView {
    fn bind_group_layout_entry(
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    fn bind_group_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(self),
        }
    }
}

impl Bindable for wgpu::Sampler {
    fn bind_group_layout_entry(
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }

    fn bind_group_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Sampler(self),
        }
    }
}
