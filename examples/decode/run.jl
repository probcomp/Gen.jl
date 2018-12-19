using Gen
using JSON

# get map from character to integer in 1..27
alphabet = collect('a':'z')
push!(alphabet, ' ')
letter_to_int = Dict{Char,Int}()
for (i, letter) in enumerate(alphabet)
    letter_to_int[letter] = i
end

# load bigram data into a transition matrix
# bigrams.json from https://gist.github.com/lydell/c439049abac2c9226e53 
data = JSON.parsefile("bigrams.json")
counts = zeros(Int64, (26, 26))
for (bigram, count) in data
    first = letter_to_int[bigram[1]]
    second = letter_to_int[bigram[2]]
    counts[second, first] = count
end
counts = counts .+ 1 # remove zeros
const probs = sum(counts, dims=1)[:] / sum(counts)
const transition = counts ./ sum(counts, dims=1)

@gen function generate_original_text(alpha::Float64, word_lengths::Vector{Int})
    my_probs = probs .* (1 - alpha) .+ fill(1 / 26., 26) .* alpha
    my_transition = transition * (1 - alpha) .+ fill(1 / 26., 26, 26) .* alpha
    local cur::Int
    for (k, len) in enumerate(word_lengths)
        cur = @addr(categorical(my_probs), (k, 1))
        for l=2:len
            cur = @addr(categorical(my_transition[:,cur]), (k, l))
        end
    end
end

@gen function model(alpha::Float64, word_lengths::Vector{Int})
    @addr(generate_original_text(alpha, word_lengths), :text)
    code = Int[@addr(uniform_discrete(1, 26), (:code, l)) for l=1:26]
    code
end

par_model = Map(model)

@gen function swap_proposal(prev_trace, m::Int)
    @addr(uniform_discrete(1, 26), :i)
    @addr(uniform_discrete(1, 26), :j)
end

function swap_bijection(input::Assignment, context)
    (model_args, proposal_args) = context
    (m,) = proposal_args
    (alphas, word_lengths_all,) = model_args
    word_lengths = word_lengths_all[1]
    output = DynamicAssignment()
    i = input[:proposal => :i]
    j = input[:proposal => :j]
    output[:proposal => :j] = i
    output[:proposal => :i] = j
    output[:model => m => (:code, i)] = input[:model => m => (:code, j)]
    output[:model => m => (:code, j)] = input[:model => m => (:code, i)]

    for (k, len) in enumerate(word_lengths)

        # update the latent words
        for l=1:len
            local curval::Int
            local newval::Int
            curval = input[:model => m => :text => (k, l)]
            if curval == i
                newval = j
            elseif curval == j
                newval = i
            else
                newval = curval
            end
            output[:model => m => :text => (k, l)] = newval
        end
    end

    (output, 0.)
end

@gen function exchange_proposal(prev_trace, m::Int)
    # exchange change m with (((m-1)+1)%n)+1
end

function exchange_bijection(input::Assignment, context)
    (model_args, proposal_args) = context
    (alphas, word_lengths) = model_args
    (m,) = proposal_args
    output = DynamicAssignment()
    n = length(alphas)
    m_plus_one = (((m-1)+1)%n)+1
    set_subassmt!(output, :model => m_plus_one, get_subassmt(input, :model => m))
    set_subassmt!(output, :model => m, get_subassmt(input, :model => m_plus_one))
    (output, 0.)
end

function to_sentence(words::Vector{Vector{Int}})
    join(map((word) -> join(alphabet[word]), words), " ")
end

function get_original_words(assmt::Assignment, word_lengths::Vector{Int})
    original_words = Vector{Vector{Int}}(undef, length(word_lengths))
    for (k, len) in enumerate(word_lengths)
        word = Vector{Int}(undef, len)
        for l=1:len
            word[l] = assmt[:text => (k, l)]
        end
        original_words[k] = word
    end
    original_words
end

function get_output_words(original_words::Vector{Vector{Int}}, code::Vector{Int})
    output_words = Vector{Vector{Int}}(undef, length(original_words))
    for (k, original_word) in enumerate(original_words)
        output_word = Vector{Int}(undef, length(original_word))
        for (l, letter) in enumerate(original_word)
            output_word[l] = code[letter]
        end
        output_words[k] = output_word
    end
    output_words
end

function do_inference(msg::Vector{T}, num_iter::Int) where {T <: AbstractString}
    assmt = DynamicAssignment()

    alphas = collect(range(0, stop=1, length=20))

    for m=1:length(alphas)
        # set initial code to the identity
        for l=1:26
            assmt[m => (:code, l)] = l
        end

        # initialize original words
        for (k, word) in enumerate(msg)
            for (l, letter) in enumerate(word)
                assmt[m => :text => (k, l)] = letter_to_int[letter]
            end
        end
    end
        
    # word lengths
    word_lengths = Vector{Int}()
    for word in msg
        push!(word_lengths, length(word))
    end

    # initial trace
    (trace, _) = initialize(par_model, (alphas, fill(word_lengths, length(alphas)),), assmt)

    # do MCMC
    for iter=1:num_iter

        # print state
        if (iter - 1) % 1 == 0
            retval = get_retval(trace)
            assmt = get_assmt(trace)
            @assert length(retval) == length(alphas)
            println()
            for m=1:length(alphas)
                code = retval[m]
                original_words = get_original_words(get_subassmt(assmt, m), word_lengths)
                output_words = get_output_words(original_words, code)
                println("$(to_sentence(original_words[1:30]))...")
            end
        end

        for m=1:length(alphas)
            (trace, accepted) = rjmcmc(trace, swap_proposal, (m,), swap_bijection)
            (trace, exchange_accepted) = rjmcmc(trace, exchange_proposal, (m,), exchange_bijection)
        end
    end
end

using Random: randperm, seed!

seed!(1) # OK
seed!(2) # OK
seed!(3) # OK
seed!(4) # OK
seed!(5) # OK
seed!(6)

original_text = """
to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
and by opposing end them to die to sleep
no more and by a sleep to say we end
the heartache and the thousand natural shocks
that flesh is heir to tis a consummation
devoutly to be wishd to die to sleep
to sleep perchance to dream ay theres the rub
for in that sleep of death what dreams may come
when we have shuffled off this mortal coil
must give us pause theres the respect
that makes calamity of so long life
for who would bear the whips and scorns of time
the oppressors wrong the proud mans contumely
the pangs of despised love the laws delay
the insolence of office and the spurns
that patient merit of the unworthy takes
when he himself might his quietus make
with a bare bodkin who would fardels bear
to grunt and sweat under a weary life
but that the dread of something after death
the undiscovered country from whose bourn
"""
original_text = join(split(original_text, "\n"), " ")
#original_text = original_text[1:20]
#println(original_text)
code = randperm(26)
original_words = map((str) -> map((letter) -> letter_to_int[letter], collect(str)), split(original_text))
output_words = map((word) -> map((letter) -> code[letter], word), original_words)
msg = map((word) -> join(alphabet[word]), output_words)
println(join(msg, " "))
do_inference(msg, 100000)
