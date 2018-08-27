import FileIO

include("model.jl")

function load_params!(gen_func, filename)
    data = FileIO.load(filename)
    for (name, param) in data
        gen_func.params[Symbol(name)] = param
    end
end

function do_inference(input_image)

    # construct trace containing observed image
    observations = DynamicChoiceTrie()
    observations["image"] = input_image

    # fill in latent variables using proposal
    latents = get_choices(simulate(proposal, (), Some(observations)))
    x = latents["x"]
    y = latents["y"]
    s = latents["size"]
    letter = latents["letter"]
    angle = latents["angle"]
    println("x: $x, y: $y, size: $s, letter: $letter, angle: $angle")

    # predicted image given latents
    (trace, _) = generate(model, (), latents)
    predicted = get_call_record(trace).retval
    predicted =  min.(ones(predicted), max.(zeros(predicted), predicted))
end

load_params!(proposal, "params.jld2")

println("do inference..")
for i=1:100
    input_image = convert(Matrix{Float64}, load("prior/img-029.png"))
    predicted = do_inference(input_image)
    output_filename = @sprintf("proposed/%03d.png", i)
    FileIO.save(output_filename, colorview(Gray, predicted))
end
