import shamrock


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_AMRGodunov(
    context = ctx, 
    vector_type = "f64_3",
    grid_repr = "i64_3")