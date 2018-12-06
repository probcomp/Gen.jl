struct SGDTrainConf
    num_batch::Int
    batch_size::Int
    num_minibatch::Int
    minibatch_size::Int
    input_extractor::Function
    constraint_extractor::Function
    minibatch_callback::Function
    batch_callback::Function
end

function sgd_train_batch(teacher::GenerativeFunction{T,U}, teacher_args::Tuple,
                         batch_student::GenerativeFunction{V,W}, conf::SGDTrainConf,
                         verbose=false) where {T,U,V,W}

    for batch=1:conf.num_batch

        # generate training batch
        training_assignment = Vector{Any}(undef, conf.batch_size)
        for i=1:conf.batch_size
            training_assignment[i] = get_assignment(simulate(teacher, teacher_args))
            if verbose && (i % 100 == 0)
                println("batch $batch, generating training data $i of $(conf.batch_size))")
            end
        end

        # train on this batch
        for minibatch=1:conf.num_minibatch
            permuted = Random.randperm(conf.batch_size)
            minibatch_idx = permuted[1:conf.minibatch_size]
            assignments = training_assignment[minibatch_idx]
            input = conf.input_extractor(assignments)
            constraints = conf.constraint_extractor(assignments)
            student_trace = assess(batch_student, input, constraints)
            avg_score = get_call_record(student_trace).score / conf.minibatch_size
            backprop_params(batch_student, student_trace, nothing)
            conf.minibatch_callback(batch, minibatch, avg_score, verbose)
        end

        conf.batch_callback(batch, verbose)
    end
end

export sgd_train_batch, SGDTrainConf
