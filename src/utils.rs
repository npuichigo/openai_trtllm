use std::fmt;
use std::marker::PhantomData;
use std::str;
use std::str::Utf8Error;

use bytes::{Buf, Bytes};
use serde::{de, Deserialize, Deserializer};

pub(crate) fn string_or_seq_string<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVec(PhantomData<Vec<String>>);

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or list of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value.to_owned()])
        }

        fn visit_seq<S>(self, visitor: S) -> Result<Self::Value, S::Error>
        where
            S: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(visitor))
        }
    }

    deserializer.deserialize_any(StringOrVec(PhantomData))
}

pub(crate) fn deserialize_bytes_tensor(encoded_tensor: Vec<u8>) -> Result<Vec<String>, Utf8Error> {
    let mut bytes = Bytes::from(encoded_tensor);
    let mut strs = Vec::new();
    while bytes.has_remaining() {
        let len = bytes.get_u32_le() as usize;
        if len <= bytes.remaining() {
            let slice = bytes.split_to(len);
            let s = str::from_utf8(&slice)?;
            strs.push(s.to_string());
        }
    }
    Ok(strs)
}
