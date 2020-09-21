using Gen

using Cairo
using FileIO
using ImageMagick
using Compat
using Compat.Base64
using Images
using ImageFiltering

const letters = String["A", "B", "C"]

###########################
# primitive distributions #
###########################

struct NoisyMatrix <: Distribution{Matrix{Float64}} end
const noisy_matrix = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64}, mu::Matrix{Float64}, noise::Float64)
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Gen.random(::NoisyMatrix, mu::Matrix{Float64}, noise::Float64)
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = mu[i, j] + randn() * noise
        end
    end
    mat
end

struct Object
    x::Float64
    y::Float64
    angle::Float64
    fontsize::Int
    letter::String
end

const width = 40
const height = 40
const min_size = 5
const max_size = 10

function render(obj::Object)
    canvas = CairoRGBSurface(width, height)
    cr = CairoContext(canvas)
    Cairo.save(cr)

    # set background color to white
    set_source_rgb(cr, 1.0, 1.0, 1.0)
    rectangle(cr, 0.0, 0.0, width, height)
    fill(cr)
    restore(cr)
    Cairo.save(cr)

    # write some letters
    set_font_face(cr, "Sans $(obj.fontsize)")
    text(cr, obj.x, obj.y, obj.letter, angle=obj.angle)

    # convert to matrix of color values
    buf = IOBuffer()
    write_to_png(canvas, buf)
    Images.Gray.(readblob(take!(buf)))
end

@gen function model()

    # object prior
    x = @trace(uniform_continuous(0, 1), "x")
    y = @trace(uniform_continuous(0, 1), "y")
    size = @trace(uniform_continuous(0, 1), "size")
    letter = letters[@trace(uniform_discrete(1, length(letters)), "letter")]
    angle = @trace(uniform_continuous(-1, 1), "angle")
    fontsize = min_size + Int(floor((max_size - min_size + 1) * size))
    object = Object(height * x, width * y, angle * 45, fontsize, letter)

    # render
    image = render(object)

    # blur it
    blur_amount = 1
    blurred = imfilter(image, Kernel.gaussian(blur_amount))

    # add speckle
    mat = convert(Matrix{Float64}, blurred)
    noise = 0.1
    @trace(noisy_matrix(mat, noise), "image")
end

const num_input = width * height
const num_hidden1 = 100
const num_hidden2 = 100
const num_output = 11

relu(x) = x .* (x .> 0)

function hidden_layer(W, b, input)
    relu(W * input + b)
end
output_layer(W, b, input) = W * input + b

function inference_network(image, W1, b1, W2, b2, W3, b3)
    h1 = hidden_layer(W1, b1, image)
    h2 = hidden_layer(W2, b2, h1)
    output_layer(W3, b3, h2)
end

@gen function proposal(prev_choices)
    @param W1::Matrix{Float64}
    @param b1::Vector{Float64}
    @param W2::Matrix{Float64}
    @param b2::Vector{Float64}
    @param W3::Matrix{Float64}
    @param b3::Vector{Float64}

    image = prev_choices["image"][:]

    # inference network
    outputs = inference_network(image, W1, b1, W2, b2, W3, b3)
    x_mu = outputs[1]
    x_std = exp.(outputs[2])
    y_mu = outputs[3]
    y_std = exp.(outputs[4])
    r_mu = exp.(outputs[5])
    r_std = exp.(outputs[6])

    # distribution over sizes of the letters
    size_alpha = exp(outputs[7])
    size_beta = exp(outputs[8])

    # distribution over 3 possible letters
    log_letter_dist = outputs[9:9 + length(letters)-1]
    letter_dist = exp.(log_letter_dist)
    letter_dist = letter_dist / sum(letter_dist)

    @trace(normal(x_mu, x_std), "x")
    @trace(normal(y_mu, y_std), "y")
    @trace(normal(r_mu, r_std), "angle")
    @trace(beta(size_alpha, size_beta), "size")
    @trace(categorical(letter_dist), "letter")
end
