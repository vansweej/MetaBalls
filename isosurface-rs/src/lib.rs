pub mod iso_field_generator;
pub(crate) mod iso_field_polygoniser;
pub(crate) mod iso_field_polygoniser2;
pub(crate) mod lookup;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
