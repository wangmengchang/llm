use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct GgufInfo {
    pub name: Option<String>,
    pub arch: Option<String>,
    pub context: Option<u64>,
}

const GGUF_MAGIC: u32 = 0x4655_4747;

pub fn read_info(path: &Path) -> Option<GgufInfo> {
    let file = File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    if len < 24 {
        return None;
    }
    let mut reader = BufReader::new(file);
    parse(&mut reader)
}

fn parse<R: Read + Seek>(reader: &mut R) -> Option<GgufInfo> {
    if read_u32(reader)? != GGUF_MAGIC {
        return None;
    }
    let version = read_u32(reader)?;
    if version < 2 {
        return None;
    }
    let _tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;

    let mut info = GgufInfo::default();
    let mut contexts: HashMap<String, u64> = HashMap::new();

    for _ in 0..kv_count.min(2048) {
        let key = read_string(reader, 512)?;

        if key.starts_with("tokenizer.") {
            break;
        }

        let value_type = read_u32(reader)?;
        match value_type {
            8 => {
                let value = read_string(reader, 4096);
                match key.as_str() {
                    "general.name" => info.name = value,
                    "general.architecture" => info.arch = value,
                    _ => {}
                }
            }
            0 | 1 | 2 | 3 | 4 | 5 | 10 | 11 => {
                let value = read_int(reader, value_type)?;
                if key.ends_with(".context_length") {
                    contexts.insert(key, value.max(0) as u64);
                }
            }
            _ => skip_value(reader, value_type, 0)?,
        }
    }

    info.context = info
        .arch
        .as_ref()
        .and_then(|arch| contexts.get(&format!("{arch}.context_length")).copied())
        .or_else(|| contexts.values().next().copied());

    Some(info)
}

fn skip_value<R: Read + Seek>(reader: &mut R, value_type: u32, depth: u32) -> Option<()> {
    if depth > 3 {
        return None;
    }
    match value_type {
        0 | 1 | 7 => {
            reader.seek(SeekFrom::Current(1)).ok()?;
        }
        2 | 3 => {
            reader.seek(SeekFrom::Current(2)).ok()?;
        }
        4..=6 => {
            reader.seek(SeekFrom::Current(4)).ok()?;
        }
        10..=12 => {
            reader.seek(SeekFrom::Current(8)).ok()?;
        }
        8 => {
            let len = read_u64(reader)?;
            reader.seek(SeekFrom::Current(len as i64)).ok()?;
        }
        9 => {
            let item_type = read_u32(reader)?;
            let len = read_u64(reader)?;
            if len > 1_000_000 {
                return None;
            }
            for _ in 0..len {
                skip_value(reader, item_type, depth + 1)?;
            }
        }
        _ => return None,
    }
    Some(())
}

fn read_int<R: Read>(reader: &mut R, value_type: u32) -> Option<i128> {
    Some(match value_type {
        0 => read_u8(reader)? as i128,
        1 => read_u8(reader)? as i8 as i128,
        2 => read_u16(reader)? as i128,
        3 => read_u16(reader)? as i16 as i128,
        4 => read_u32(reader)? as i128,
        5 => read_u32(reader)? as i32 as i128,
        10 => read_u64(reader)? as i128,
        11 => read_u64(reader)? as i64 as i128,
        _ => return None,
    })
}

fn read_string<R: Read + Seek>(reader: &mut R, max_len: u64) -> Option<String> {
    let len = read_u64(reader)?;
    if len > max_len {
        reader.seek(SeekFrom::Current(len as i64)).ok()?;
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).trim().to_string())
}

fn read_u8<R: Read>(reader: &mut R) -> Option<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).ok()?;
    Some(buf[0])
}

fn read_u16<R: Read>(reader: &mut R) -> Option<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf).ok()?;
    Some(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> Option<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).ok()?;
    Some(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Option<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).ok()?;
    Some(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn push_header(out: &mut Vec<u8>, kv_count: u64) {
        out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&kv_count.to_le_bytes());
    }

    #[test]
    fn reads_architecture_name_and_context() {
        let mut out = Vec::new();
        push_header(&mut out, 4);

        push_string(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes());
        push_string(&mut out, "qwen3");

        push_string(&mut out, "general.name");
        out.extend_from_slice(&8u32.to_le_bytes());
        push_string(&mut out, "Qwen3 8B");

        push_string(&mut out, "qwen3.context_length");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&32768u32.to_le_bytes());

        push_string(&mut out, "general.file_type");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&15u32.to_le_bytes());

        let info = parse(&mut Cursor::new(out)).unwrap();
        assert_eq!(info.arch.as_deref(), Some("qwen3"));
        assert_eq!(info.name.as_deref(), Some("Qwen3 8B"));
        assert_eq!(info.context, Some(32768));
    }

    #[test]
    fn rejects_non_gguf_data() {
        let data = b"not a model".to_vec();
        assert!(parse(&mut Cursor::new(data)).is_none());
    }
}
