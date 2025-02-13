def minimize_prompt(model, tokenizer, input_str, target_str, system_prompt, chat_template, device, optimization_args,
                    max_tokens=30, max_failure_limit=None):
    # Initialize based on target string length
    target_tokens = len(tokenizer.encode(target_str))
    n_tokens_in_prompt = target_tokens
    running_min = 0
    running_max = max_tokens if max_tokens != -1 else target_tokens * 5
    max_failed = 0
    min_success = 0
    success = False
    best_prompt = None
    done = False
    best_slices = (None, None, None, None)
    
    # Set failure limit based on target length
    if max_failure_limit is None:
        max_failure_limit = target_tokens * 2
    
    # Set initial num_steps based on tokens/10 brackets
    base_steps = 200
    num_brackets = (n_tokens_in_prompt // 10)
    current_steps = int(base_steps * (1.2 ** num_brackets))
    optimization_args["num_steps"] = current_steps

    while not done:
        logging.info("\n------------------------------------\n")
        logging.info(f"{n_tokens_in_prompt} tokens in the prompt (max_failed={max_failed}, min_success={min_success})")
        logging.info(f"Using {current_steps} optimization steps")
        
        # Check if we've exceeded the failure limit
        if max_failed >= max_failure_limit:
            logging.info(f"Early termination: max_failed ({max_failed}) exceeded limit ({max_failure_limit})")
            done = True
            continue
        
        # Ensure we don't exceed maximum tokens
        if n_tokens_in_prompt > running_max:
            n_tokens_in_prompt = running_max
        
        input_ids, free_token_slice, input_slice, target_slice, loss_slice = prompt_opt.prep_text(
            input_str,
            target_str,
            tokenizer,
            system_prompt,
            chat_template,
            n_tokens_in_prompt,
            device
        )

        # Run optimization with current token length
        if optimization_args["discrete_optimizer"] == "gcg":
            solution = prompt_opt.optimize_gcg(
                model, input_ids, input_slice, free_token_slice, target_slice,
                loss_slice, current_steps,
                batch_size=optimization_args["batch_size"],
                topk=optimization_args["topk"],
                mini_batch_size=optimization_args["mini_batch_size"]
            )
        elif optimization_args["discrete_optimizer"] == "random_search":
            solution = prompt_opt.optimize_random_search(
                model, input_ids, input_slice, free_token_slice,
                target_slice, loss_slice, current_steps,
                batch_size=optimization_args["batch_size"],
                mini_batch_size=optimization_args["mini_batch_size"]
            )
        else:
            raise ValueError("discrete_optimizer must be one of ['gcg', 'random_search']")

        target_acquired = prompt_opt.check_output_with_hard_tokens(
            model, 
            solution["input_ids"].unsqueeze(0),
            target_slice,
            loss_slice
        )

        if target_acquired:
            logging.info(f"Target acquired with {n_tokens_in_prompt} tokens in the prompt")
            success = True
            best_prompt = solution["input_ids"]
            best_slices = (free_token_slice, input_slice, target_slice, loss_slice)
            min_success = n_tokens_in_prompt
            
            # Binary search: Try halfway between max_failed and current tokens
            new_num_tokens = max_failed + (n_tokens_in_prompt - max_failed) // 2
            logging.info(f"SUCCESS: Setting min_success={n_tokens_in_prompt}. Trying {new_num_tokens} next")
            
        else:
            logging.info(f"Target NOT acquired with {n_tokens_in_prompt} tokens in the prompt")
            max_failed = n_tokens_in_prompt
            
            if n_tokens_in_prompt < min_success and min_success > 0:
                # Binary search between max_failed and min_success
                new_num_tokens = max_failed + (min_success - max_failed) // 2
                logging.info(f"FAIL: Setting max_failed={n_tokens_in_prompt}. Trying {new_num_tokens} next")
            else:
                # Double tokens when no success found yet or current >= min_success
                new_num_tokens = n_tokens_in_prompt * 2
                logging.info(f"FAIL: Setting max_failed={n_tokens_in_prompt}. Doubling to {new_num_tokens}")

        # Ensure we test all values when gap is small
        gap = min_success - max_failed if min_success > 0 and max_failed > 0 else float('inf')
        if gap > 1 and gap <= 3:
            new_num_tokens = max_failed + 1
            logging.info(f"Small gap detected ({gap}). Testing next value: {new_num_tokens}")

        # Update num_steps for next iteration based on new token count
        new_brackets = (new_num_tokens // 10)
        current_steps = int(base_steps * (1.2 ** new_brackets))
        optimization_args["num_steps"] = current_steps

        # Check termination conditions
        if (new_num_tokens >= running_max or
            new_num_tokens <= running_min or
            (min_success > 0 and max_failed > 0 and (min_success - max_failed) <= 1)):
            done = True
        else:
            n_tokens_in_prompt = new_num_tokens

    output = {
        "free_token_slice": best_slices[0] if best_slices[0] is not None else free_token_slice,
        "input_slice": best_slices[1] if best_slices[1] is not None else input_slice,
        "target_slice": best_slices[2] if best_slices[2] is not None else target_slice,
        "loss_slice": best_slices[3] if best_slices[3] is not None else loss_slice,
        "success": success,
        "num_free_tokens": min_success if success else None,
        "input_ids": best_prompt,
    }
    return output