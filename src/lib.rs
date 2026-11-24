use std::io::{Cursor, Write};

use serde::Deserialize;
use serde_json::{Value, json};
use zip::{ZipWriter, write::SimpleFileOptions};

pub const DESIGN_PROMPT: &str = include_str!("prompt.txt");
const FACE_NAMES: [&str; 6] = ["north", "south", "east", "west", "up", "down"];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid design JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid character design: {0}")]
    Design(&'static str),
    #[error("texture encoding failed: {0}")]
    Png(#[from] png::EncodingError),
    #[error("archive failed: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("archive write failed: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Design {
    name: String,
    cubes: Vec<Cube>,
}

#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Bone {
    Body,
    Head,
    RightArm,
    LeftArm,
    RightLeg,
    LeftLeg,
}

impl Bone {
    const ALL: [Self; 6] = [
        Self::Body,
        Self::Head,
        Self::RightArm,
        Self::LeftArm,
        Self::RightLeg,
        Self::LeftLeg,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Body => "body",
            Self::Head => "head",
            Self::RightArm => "right_arm",
            Self::LeftArm => "left_arm",
            Self::RightLeg => "right_leg",
            Self::LeftLeg => "left_leg",
        }
    }

    fn pivot(self) -> [i32; 3] {
        match self {
            Self::Body => [0, 12, 0],
            Self::Head => [0, 24, 0],
            Self::RightArm => [-5, 22, 0],
            Self::LeftArm => [5, 22, 0],
            Self::RightLeg => [-2, 12, 0],
            Self::LeftLeg => [2, 12, 0],
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Cube {
    bone: Bone,
    origin: [f32; 3],
    size: [f32; 3],
    color: String,
}

fn color_rgb(value: &str) -> Result<[u8; 3], Error> {
    let hex = value
        .strip_prefix('#')
        .filter(|v| v.len() == 6 && v.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or(Error::Design("color must be #RRGGBB"))?;
    let rgb = u32::from_str_radix(hex, 16).map_err(|_| Error::Design("invalid color"))?;
    let [_, r, g, b] = rgb.to_be_bytes();
    Ok([r, g, b])
}

fn parse_design(reply: &str) -> Result<Design, Error> {
    let text = reply.trim();
    let text = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .and_then(|v| v.trim().strip_suffix("```"))
        .unwrap_or(text);
    let design: Design = serde_json::from_str(text)?;
    if design.name.trim().is_empty()
        || design.name.chars().count() > 80
        || !(6..=64).contains(&design.cubes.len())
    {
        return Err(Error::Design("expected a display name and 6 to 64 cubes"));
    }
    if Bone::ALL
        .iter()
        .any(|bone| !design.cubes.iter().any(|cube| cube.bone == *bone))
    {
        return Err(Error::Design("all six animated bones must have geometry"));
    }
    for cube in &design.cubes {
        if cube.origin.iter().any(|x| !(-96.0..=96.0).contains(x))
            || cube.size.iter().any(|x| !(0.1..=96.0).contains(x))
        {
            return Err(Error::Design("cube origin or size is out of range"));
        }
        color_rgb(&cube.color)?;
    }
    Ok(design)
}

const ATLAS_WIDTH: u32 = 32;
const ATLAS_HEIGHT: u32 = 32;
const TILE_SIZE: usize = 4;

fn tile(index: usize) -> [usize; 2] {
    [index % 8 * TILE_SIZE, index / 8 * TILE_SIZE]
}

fn texture(design: &Design) -> Result<Vec<u8>, Error> {
    let mut pixels = vec![0; (ATLAS_WIDTH * ATLAS_HEIGHT * 3) as usize];
    for (index, cube) in design.cubes.iter().enumerate() {
        let [x, y] = tile(index);
        let rgb = color_rgb(&cube.color)?;
        for dy in 0..TILE_SIZE {
            for dx in 0..TILE_SIZE {
                let offset = ((y + dy) * ATLAS_WIDTH as usize + x + dx) * 3;
                pixels[offset..offset + 3].copy_from_slice(&rgb);
            }
        }
    }
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, ATLAS_WIDTH, ATLAS_HEIGHT);
    encoder.set_color(png::ColorType::Rgb);
    encoder.write_header()?.write_image_data(&pixels)?;
    Ok(bytes)
}

fn geometry(design: &Design, arm_only: bool) -> Value {
    let bones: Vec<_> = Bone::ALL
        .into_iter()
        .filter(|bone| !arm_only || matches!(bone, Bone::LeftArm | Bone::RightArm))
        .map(|bone| {
            let cubes: Vec<_> = design
                .cubes
                .iter()
                .enumerate()
                .filter(|(_, cube)| cube.bone == bone)
                .map(|(index, cube)| {
                    let uv: serde_json::Map<_, _> = FACE_NAMES
                        .into_iter()
                        .map(|face| {
                            (
                                face.into(),
                                json!({
                                    "uv": tile(index), "uv_size": [TILE_SIZE, TILE_SIZE],
                                }),
                            )
                        })
                        .collect();
                    json!({"origin": cube.origin, "size": cube.size, "uv": uv})
                })
                .collect();
            let mut entry = json!({"name": bone.name(), "pivot": bone.pivot(), "cubes": cubes});
            if !arm_only && !matches!(bone, Bone::Body | Bone::LeftLeg | Bone::RightLeg) {
                entry["parent"] = json!("body");
            }
            entry
        })
        .collect();
    let identifier = if arm_only {
        "geometry.generated_arm"
    } else {
        "geometry.generated"
    };
    json!({"format_version":"1.12.0","minecraft:geometry":[{"description":{
        "identifier":identifier,"texture_width":ATLAS_WIDTH,"texture_height":ATLAS_HEIGHT
    },"bones":bones}]})
}

fn swing(amplitude: i32, period: f32) -> Value {
    let rotations = [
        ("left_arm", -amplitude),
        ("right_arm", amplitude),
        ("left_leg", amplitude),
        ("right_leg", -amplitude),
    ];
    let bones: serde_json::Map<_, _> = rotations
        .into_iter()
        .map(|(bone, angle)| {
            (
                bone.into(),
                json!({"rotation":{
                    "0.0":[angle,0,0],
                    format!("{:.3}",period / 2.0):[-angle,0,0],
                    format!("{period:.3}"):[angle,0,0]
                }}),
            )
        })
        .collect();
    json!({"loop":true,"animation_length":period,"bones":bones})
}

/// Turn a structured text-model reply into a plain, editable YSM Spec 2 ZIP.
pub fn package_design(reply: &str) -> Result<Vec<u8>, Error> {
    let design = parse_design(reply)?;
    let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
    let manifest = json!({"spec":2,"metadata":{"name":design.name},"properties":{
        "height_scale":1.0,"width_scale":1.0,"default_texture":"textures/default.png"
    },"files":{"player":{
        "model":{"main":"models/main.json","arm":"models/arm.json"},
        "animation":{"main":"animations/main.animation.json","arm":"animations/arm.animation.json"},
        "texture":[{"uv":"textures/default.png"}]
    }}});
    let idle = json!({"loop":true,"animation_length":2.0,"bones":{
        "body":{"position":{"0.0":[0,0,0],"1.0":[0,0.12,0],"2.0":[0,0,0]}}
    }});
    let main_animation = json!({"format_version":"1.8.0","animations":{
        "idle":idle,"walk":swing(28,0.8),"run":swing(48,0.48)
    }});
    let arm_animation =
        json!({"format_version":"1.8.0","animations":{"idle":{"loop":true,"bones":{}}}});
    for (path, body) in [
        ("ysm.json", manifest),
        ("models/main.json", geometry(&design, false)),
        ("models/arm.json", geometry(&design, true)),
        ("animations/main.animation.json", main_animation),
        ("animations/arm.animation.json", arm_animation),
    ] {
        archive.start_file(path, SimpleFileOptions::default())?;
        serde_json::to_writer(&mut archive, &body)?;
    }
    archive.start_file("textures/default.png", SimpleFileOptions::default())?;
    archive.write_all(&texture(&design)?)?;
    Ok(archive.finish()?.into_inner())
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    #[test]
    fn archive_contains_generated_geometry_and_texture() {
        let cubes: Vec<_> = Bone::ALL
            .into_iter()
            .map(|bone| {
                json!({
                    "bone":bone.name(),"origin":[1,2,3],"size":[4,5,6],"color":"#AABBCC"
                })
            })
            .collect();
        let design = json!({"name":"Example","cubes":cubes});
        let bytes = package_design(&design.to_string()).unwrap();
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        let model: Value =
            serde_json::from_reader(zip.by_name("models/main.json").unwrap()).unwrap();
        assert_eq!(
            model["minecraft:geometry"][0]["bones"][0]["cubes"][0]["origin"],
            json!([1., 2., 3.])
        );
        let animation: Value =
            serde_json::from_reader(zip.by_name("animations/main.animation.json").unwrap())
                .unwrap();
        assert!(
            animation["animations"]["walk"]["bones"]["left_leg"]["rotation"]["0.400"].is_array()
        );
        let mut signature = [0; 8];
        zip.by_name("textures/default.png")
            .unwrap()
            .read_exact(&mut signature)
            .unwrap();
        assert_eq!(&signature, b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn rejects_model_without_core_bones() {
        assert!(matches!(
            package_design(r#"{"name":"Example","cubes":[]}"#),
            Err(Error::Design(_))
        ));
    }
}
