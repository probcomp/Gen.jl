using Gen
using JSON

# get map from character to integer in 1..26
alphabet = collect('a':'z')
push!(alphabet, ' ')
letter_to_int = Dict{Char,Int}()
for (i, letter) in enumerate(alphabet)
    letter_to_int[letter] = i
end

# load bigram data into a transition matrix
# bigrams.json from https://gist.github.com/lydell/c439049abac2c9226e53 
data = JSON.parsefile("$(@__DIR__)/bigrams.json")
counts = zeros(Int64, (27, 27))
for (bigram, count) in data
    first = letter_to_int[bigram[1]]
    second = letter_to_int[bigram[2]]
    counts[second, first] = count
end
counts = counts .+ 1 # remove zeros
start_counts = [counts[i,27] for i=1:27]
const probs = start_counts / sum(start_counts)
const transition = counts ./ sum(counts, dims=1)

@gen function generate_original_text(alpha::Float64, len::Int)
    my_probs = probs .* (1 - alpha) .+ fill(1 / 27., 27) .* alpha
    my_transition = transition * (1 - alpha) .+ fill(1 / 27., 27, 27) .* alpha
    local cur::Int
    cur = @addr(categorical(my_probs), 1)
    for l=2:len
        cur = @addr(categorical(my_transition[:,cur]), l)
    end
end

@gen function model(alpha::Float64, len::Int)
    @addr(generate_original_text(alpha, len), :text)
    code = Int[@addr(uniform_discrete(1, 27), (:code, l)) for l=1:27]
    code
end

par_model = Map(model)

@gen function swap_proposal(prev_trace, m::Int)
    @addr(uniform_discrete(1, 27), :i)
    @addr(uniform_discrete(1, 27), :j)
end

function swap_involution(trace, fwd_assmt::Assignment, fwd_ret, proposal_args::Tuple)
    assmt = get_assmt(trace)
    model_args = get_args(trace)
    (replica,) = proposal_args
    (alphas, len_repeated) = model_args
    len::Int = len_repeated[1]
    bwd_assmt = DynamicAssignment()
    i = fwd_assmt[:i]
    j = fwd_assmt[:j]
    bwd_assmt[:j] = i
    bwd_assmt[:i] = j
    constraints = DynamicAssignment()
    constraints[replica => (:code, i)] = assmt[replica => (:code, j)]
    constraints[replica => (:code, j)] = assmt[replica => (:code, i)]

    # update the latent letters
    for l=1:len
        local cur_char::Int
        local new_char::Int
        cur_char = assmt[replica => :text => l]
        if cur_char == i
            new_char = j
        elseif cur_char == j
            new_char = i
        else
            new_char = cur_char
        end
        constraints[replica => :text => l] = new_char
    end

    (new_trace, weight, _, _) = force_update(model_args, noargdiff, trace, constraints)
    (new_trace, bwd_assmt, weight)
end

@gen function exchange_proposal(prev_trace, m::Int)
    # exchange change m with (((m-1)+1)%n)+1
end

function exchange_involution(trace, fwd_assmt::Assignment, fwd_ret, proposal_args::Tuple)
    assmt = get_assmt(trace)
    model_args = get_args(trace)
    (alphas, _) = model_args
    (replica,) = proposal_args
    constraints = DynamicAssignment()
    num_replicas = length(alphas)
    replica_plus_one = (((replica-1)+1)%num_replicas)+1
    set_subassmt!(constraints, replica_plus_one, get_subassmt(assmt, replica))
    set_subassmt!(constraints, replica, get_subassmt(assmt, replica_plus_one))
    (new_trace, weight, _, _) = force_update(model_args, noargdiff, trace, constraints)
    (new_trace, EmptyAssignment(), weight)
end

to_string(text::Vector{Int}) = join(alphabet[text])

function get_original_text(assmt::Assignment, len::Int)
    Int[assmt[:text => l] for l=1:len]
end

function get_output_text(original_text::Vector{Int}, code::Vector{Int})
    code[original_text]
end

function do_inference(encoded_text::AbstractString, num_iter::Int)
    len = length(encoded_text)
    assmt = DynamicAssignment()

    alphas = collect(range(0, stop=1, length=10))
    num_replicas = length(alphas)

    for replica=1:num_replicas
        # set initial code to the identity
        for l=1:27
            assmt[replica => (:code, l)] = l
        end

        # initialize original text 
        for (l, char) in enumerate(encoded_text)
            assmt[replica => :text => l] = letter_to_int[char]
        end
    end
        
    # initial trace
    (trace, _) = initialize(par_model, (alphas, fill(len, num_replicas)), assmt)

    # do MCMC
    for iter=1:num_iter

        # print state
        if (iter - 1) % 1 == 0
            retval = get_retval(trace)
            assmt = get_assmt(trace)
            @assert length(retval) == length(alphas)
            println()
            println()
            for replica=1:num_replicas
                code = retval[replica]
                original_text = get_original_text(get_subassmt(assmt, replica), len)
                output_text = get_output_text(original_text, code)
                println("$(to_string(original_text[1:120]))...")
            end
        end

        for replica=1:num_replicas
            (trace, _) = metropolis_hastings(trace, swap_proposal, (replica,), swap_involution)
            (trace, _) = metropolis_hastings(trace, exchange_proposal, (replica,), exchange_involution)
        end
    end
end

using Random: randperm, seed!

seed!(1)
seed!(2)
seed!(3)

original_text = join(split("""
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
""", "\n"), " ")

println("original text:")
println(original_text)
code = randperm(27)
original_text_int = map((char) -> letter_to_int[char], collect(original_text))
encoded_text = join(map((letter_int) -> alphabet[letter_int], code[original_text_int]))
println("encoded text:")
println(encoded_text)
do_inference(encoded_text, 100000)
