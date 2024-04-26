import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
import tiktoken
import openai

import traceback
import os
import ast

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
    

    
def create_prompt_few_shot_learning(examples, language, new_source_code, prompt_template):
        def create_demonstration(examples):
            demonstrations = ""
            
            for index, example in enumerate(examples):
                
                # Determine analysis status
                #analysis_explanation = example['chain-of-thought']
                
                if example['target'] == 1:
                    analysis_status = "Vulnerable"
                else:
                    analysis_status = "Non-Vulnerable"
                
                # if example['target'] == 1:
                #     analysis_status = "Vulnerable"
                    
                #     flaw_lines = example['flaw_line'].split('/~/')
                #     flaw_indices = example['flaw_line_index'].split(',')
                    
                #     # Combining flaw lines and their respective indices into the desired format
                #     combined_flaws = [
                #         f"Line: {index}, Vulnerable Code: {line}" 
                #         for index, line in zip(flaw_indices, flaw_lines)
                #     ]
                #     vulnerable_lines = "\n".join(combined_flaws)
                # else:
                #     analysis_status = "Non-Vulnerable"
                #     vulnerable_lines = "No vulnerabilities are found"
                
                
                
                demonstration = prompt_template['few_shot_learning_template']['code_demonstration_template'].format(
                    func_before=example['processed_func'], 
                    analysis=analysis_status,
                    i=str(index+1))
                demonstrations += demonstration + "\n"
            return demonstrations
        
        code_demonstrations = create_demonstration(examples)
        
        prompt = prompt_template['few_shot_learning_template']['prompt_template'].format(
            language=language,
            new_source_code=new_source_code,
            code_demonstrations=code_demonstrations
        )
        
        return prompt
    
def predict_vulnerability(prompt_template, new_source_code, language, model="gpt-3.5-turbo", 
                          prompt_printing_flg=False, prompt=None):
    
        # Generate the prompt
        if prompt is None:
            prompt = prompt_template['testing_template'].format(new_source_code=new_source_code, 
                                                                language=language)
        
        # System message
        system_content = prompt_template['system_content_template'].format(language=language)
        
        # Messages for GPT API
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        # Predict using OpenAI Chat API
        response = chat_completion_with_backoff(
            model=model,
            messages=messages,
            temperature=0,
        )

        # Extract the prediction
        function_level_prediction = response.choices[0].message.content.strip()
        
        # if function level is vulnerable, predict the line-level function
        analysis = extract_analysis(function_level_prediction)

        # Line-level prediction
        if analysis.strip().lower() == "vulnerable":
            messages.append({"role": "assistant", "content": function_level_prediction})
            messages.append({"role": "user", "content": prompt_template['line_level_template']})
            
            # Predict using OpenAI Chat API
            response = chat_completion_with_backoff(
                model=model,
                messages=messages,
                temperature=0,
            )
            line_level_prediction = response.choices[0].message.content.strip()
        else:
            line_level_prediction = ""

        if prompt_printing_flg:
            print(prompt)
            print("---")
            print(function_level_prediction)
            print("***")
        
        return function_level_prediction, prompt, line_level_prediction
    
def execute_prompt(prompt, language, model="gpt-3.5-turbo"):

    # System message
    system_content = f"You are an expert {language} programmer"
        
    # Messages for GPT API
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
        
    # Predict using OpenAI Chat API
    response = chat_completion_with_backoff(
        model=model,
        messages=messages,
        temperature=0,
    )

    # Extract the prediction
    result = response.choices[0].message.content.strip()
    
    return result
    
def upload_fine_tuning_file(fine_tuning_file):
    return openai.File.create(
            file=open(fine_tuning_file, "rb"),
            purpose='fine-tune'
        )    

def create_fine_tuning_job(training_file, model):
    return openai.FineTuningJob.create(training_file=training_file, model=model)

def retrieve_job(job_id):
    return openai.FineTuningJob.retrieve(job_id)
    
def check_fine_tuning_data_path(data_path):

        # Load dataset
        with open(data_path) as f:
            dataset = [json.loads(line) for line in f]

        # We can inspect the data quickly by checking the number of examples and the first item

        # Initial dataset stats
        print("Num examples:", len(dataset))
        print("First example:")
        for message in dataset[0]["messages"]:
            print(message)

        # Now that we have a sense of the data, we need to go through all the different examples and check to make sure the formatting is correct and matches the Chat completions message structure

        # Format error checks
        format_errors = defaultdict(int)

        for ex in dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1

                if any(k not in ("role", "content", "name") for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in ("system", "user", "assistant"):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                if not content or not isinstance(content, str):
                    format_errors["missing_content"] += 1

            if not any(message.get("role", None) == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

        if format_errors:
            print("Found errors:")
            for k, v in format_errors.items():
                print(f"{k}: {v}")
        else:
            print("No errors found")

        # Beyond the structure of the message, we also need to ensure that the length does not exceed the 4096 token limit.

        # Token counting functions
        encoding = tiktoken.get_encoding("cl100k_base")

        # not exact!
        # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3
            return num_tokens

        def num_assistant_tokens_from_messages(messages):
            num_tokens = 0
            for message in messages:
                if message["role"] == "assistant":
                    num_tokens += len(encoding.encode(message["content"]))
            return num_tokens

        def print_distribution(values, name):
            print(f"\n#### Distribution of {name}:")
            print(f"min / max: {min(values)}, {max(values)}")
            print(f"mean / median: {np.mean(values)}, {np.median(values)}")
            print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

        # Last, we can look at the results of the different formatting operations before proceeding with creating a fine-tuning job:

        # Warnings and tokens counts
        n_missing_system = 0
        n_missing_user = 0
        n_messages = []
        convo_lens = []
        assistant_message_lens = []

        for ex in dataset:
            messages = ex["messages"]
            if not any(message["role"] == "system" for message in messages):
                n_missing_system += 1
            if not any(message["role"] == "user" for message in messages):
                n_missing_user += 1
            n_messages.append(len(messages))
            convo_lens.append(num_tokens_from_messages(messages))
            assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

        print("Num examples missing system message:", n_missing_system)
        print("Num examples missing user message:", n_missing_user)
        print_distribution(n_messages, "num_messages_per_example")
        print_distribution(convo_lens, "num_total_tokens_per_example")
        print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
        n_too_long = sum(l > 4096 for l in convo_lens)
        print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

        # Pricing and default n_epochs estimate
        MAX_TOKENS_PER_EXAMPLE = 4096

        MIN_TARGET_EXAMPLES = 100
        MAX_TARGET_EXAMPLES = 25000
        TARGET_EPOCHS = 3
        MIN_EPOCHS = 1
        MAX_EPOCHS = 25

        n_epochs = TARGET_EPOCHS
        n_train_examples = len(dataset)
        if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
            n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
        elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
            n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

        n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
        print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
        print(f"By default, you'll train for {n_epochs} epochs on this dataset")
        print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
        print("See pricing page to estimate total costs")
        
def predict_with_template(record, model, examples, language, prompt_template):
        # Predict
        new_source_code = record[1]
        
        # If few-shot is True, generate prompt with few-shot learning. Otherwise, using testing prompt template
        if examples is not None:
            prompt = create_prompt_few_shot_learning(
                examples=examples,
                language=language,
                new_source_code=new_source_code,
                prompt_template=prompt_template)
            
            gpt_response, prompt, line_level_prediction = predict_vulnerability(
                prompt_template=prompt_template,
                new_source_code=new_source_code, 
                prompt_printing_flg=False, 
                model=model, 
                language=language, 
                prompt=prompt)
        else:
            gpt_response, prompt, line_level_prediction = predict_vulnerability(
                prompt_template=prompt_template,
                new_source_code=new_source_code, 
                prompt_printing_flg=False, 
                model=model, 
                language=language)
            
        # Extract analysis from the response
        gpt_analysis = extract_analysis(gpt_response.lower().strip())
            
        # Get 0 or 1 based on gpt_analysis
        gpt = 0 if "non-vulnerable" in gpt_analysis else 1

        return gpt, gpt_response, prompt, line_level_prediction

def evaluate_gpt(language,
                 model,
                 dataset,
                 prompt_template,
                 examples=None,
                 file_output=None,
                 save_prompt=True):
    
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    print("Model:", model)
        
    # check language
    language = language.lower()
        
    print(f"Number of test records: {len(dataset)}")
    print(dataset.value_counts("target"))
    
    # Evaluation
    test_dataset = dataset.copy()
    
    actuals = test_dataset['target']
    gpt_predictions = []
    gpt_func_level_prediction = []
    gpt_prompt = []
    gpt_line_level_prediction = []
    
    for record in tqdm(test_dataset.itertuples(), total=len(test_dataset), desc='Predicting', unit='record'):
        try:
            gpt, gpt_response, prompt, line_level_prediction = predict_with_template(record=record, 
                                                              model=model, 
                                                              examples=examples, 
                                                              language=language, 
                                                              prompt_template=prompt_template)
        except Exception as e1:
            print(e1)
                
            gpt = 1
            gpt_response = "Limit token exceeded"
            line_level_prediction = ""
            prompt = ""
            tqdm.write(f"Exception: {e1}")
                
        # Append to predictions
        gpt_predictions.append(gpt)
        gpt_func_level_prediction.append(gpt_response)
        gpt_prompt.append(prompt)
        gpt_line_level_prediction.append(line_level_prediction)
    
    # Calculate metrics    
    gpt_precision, gpt_recall, gpt_f1_score, gpt_accuracy, mcc = calculate_func_metrics(gpt_predictions, actuals)

    print(f"GPT F1 Score: {gpt_f1_score}")
    print(f"GPT Precision: {gpt_precision}")
    print(f"GPT Recall: {gpt_recall}")
    print(f"GPT Accuracy: {gpt_accuracy*100}%")
    print(f"GPT MCC: {mcc}")
    
    # Save to files
    test_dataset['GPT Prediction'] = gpt_predictions
    test_dataset['GPT Func Level'] = gpt_func_level_prediction
    test_dataset['GPT Line Level'] = gpt_line_level_prediction
    test_dataset['Prompt'] = gpt_prompt
    
    # File output
    if file_output is None:
        if examples is not None:
            number_of_examples = len(examples)
            file_output = f"result/{language}/gpt_few_shot_{number_of_examples}.csv"
        elif model != "gpt-3.5-turbo":
            file_output = f"result/{language}/gpt_fine_tuning.csv"
        else:
            file_output = f"result/{language}/gpt_default.csv"
            
    ensure_dir(file_output)
    
    # Check save prompt
    if save_prompt:
        test_dataset.to_csv(file_output, index=False)
    else:
        test_dataset.drop(columns=['Prompt']).to_csv(file_output, index=False)
    
    return test_dataset


def calculate_func_metrics(predictions, actuals):
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    accuracy = accuracy_score(actuals, predictions)
    
    # Matthews correlation coefficient (MCC)
    mcc = matthews_corrcoef(actuals, predictions)
    
    return precision, recall, f1, accuracy, mcc
    
def extract_analysis(string):
    start_marker = "analysis:"
    end_marker = "\n"

    start_pos = string.lower().find(start_marker) + len(start_marker)
    
    # Find the end_marker after start_pos
    end_pos = string.lower().find(end_marker, start_pos)

    # If end_marker is found, slice the string up to end_pos; otherwise, slice to the end of the string
    analysis_value = string[start_pos:end_pos] if end_marker in string and end_pos != -1 else string[start_pos:]

    return analysis_value.strip()
    
def calculate_func_metrics_from_file(file_path, is_gpt=True, print_data=True):
        df = pd.read_csv(file_path)

        if is_gpt:
            prediction_column = 'GPT Prediction'
            actual_column = 'target'
        else:
            prediction_column = 'y_preds'
            actual_column = 'y_trues'
        
        predictions = df[prediction_column]
        actuals = df[actual_column]

        precision, recall, f1, accuracy, mcc = calculate_func_metrics(predictions, actuals)
        if f1 == 0:
            print(f"Zero F1: {file_path}")
        if print_data:
            print(f"\tF1 Score: {f1}")
            print(f"\tPrecision: {precision}")
            print(f"\tRecall: {recall}")
            print(f"\tAccuracy: {accuracy * 100}%")
            print(f"\tMCC: {mcc}")
        
        return precision, recall, f1, accuracy, mcc

def calculate_line_metrics_from_file(file_path, total_lines, total_vulnerable_lines, overlap, is_gpt=True, print_data=True):
    
        df = pd.read_csv(file_path)

        if is_gpt:
            prediction_column = 'GPT Prediction'
            actual_column = 'target'
        else:
            prediction_column = 'y_preds'
            actual_column = 'y_trues'
        
        predictions = df[prediction_column]
        actuals = df[actual_column]
        
        # Line level metrics
        line_level_actuals = df['flaw_line_index']
        
        if is_gpt: 
            line_level_predictions = df['GPT Line Level']
        else:
            line_level_predictions = df['LineVul Line Level']
            
            
        processed_func = df['processed_func']
            
        top_1_acc, top_3_acc, top_5_acc, top_10_acc, ifa, effort_at_20_recall, recall_at_one_percent_loc = calculate_line_metrics(actuals, predictions, line_level_actuals, line_level_predictions, is_gpt, processed_func, overlap=overlap, total_lines=total_lines, total_vulnerable_lines=total_vulnerable_lines)
        
        if print_data:
            print(f"\tTop 1 Accuracy: {top_1_acc * 100}%")
            print(f"\tTop 3 Accuracy: {top_3_acc * 100}%")
            print(f"\tTop 5 Accuracy: {top_5_acc * 100}%")
            print(f"\tTop 10 Accuracy: {top_10_acc * 100}%")
            print(f"\tIFA: {ifa}")
            
            print(f"\teffort_at_20_recall: {effort_at_20_recall}")
            print(f"\trecall_at_one_percent_loc: {recall_at_one_percent_loc}")
        
        return top_1_acc, top_3_acc, top_5_acc, top_10_acc, ifa, effort_at_20_recall, recall_at_one_percent_loc
    
def calculate_line_metrics(actuals, predictions, line_level_actuals, line_level_predictions, is_gpt, processed_func, overlap, total_lines, total_vulnerable_lines):
    top_10_accuracies = []
    top_3_accuracies = []
    top_5_accuracies = []
    top_1_accuracies = []
    total_min_clean_lines_inspected = 0
    total_functions = 0
    
    twenty_percent_vuln_lines = int(total_vulnerable_lines * 0.2)
    one_percent_loc = int(total_lines * 0.01)
    
    # Initialize counters
    effort_to_find_vuln = 0
    correctly_identified_vuln_lines = 0
    vuln_lines_in_top_one_percent = 0
    
    if overlap is None:
        overlap = processed_func.tolist()
    
    for actual, prediction, ground_truth_str, tp_prediction_strings, source_code in zip(actuals, predictions, line_level_actuals, line_level_predictions, processed_func):
        actual = int(actual)
        prediction = int(prediction)
        tp_predictions = None
        
                    
        
        # print(f"(Actual, Prediction) = ({actual}, {prediction})")
        # Only proceed if both actuals and predictions indicate vulnerability (TP)
        if actual == 1 and prediction == 1 and source_code in overlap:
             # Convert the ground_truth_str to a list of integers
            if type(ground_truth_str) is not str:
                ground_truth_str = str(ground_truth_str)
            ground_truth = [int(float(x.strip())) for x in ground_truth_str.split(",")]

            if is_gpt:
                try:
                    tp_predictions = get_vulnerable_line_numbers(tp_prediction_strings, source_code)
                except:
                    print(f"tp_prediction_strings: {tp_prediction_strings}")
                    print(f"source code: {source_code}")
                    
            else:
                tp_predictions = ast.literal_eval(tp_prediction_strings)
            
            loc = len(source_code.split("\n"))
            #print("loc = ", loc)
            #print("tp_predictions", tp_predictions)
            
            total_functions += 1

            # Check if any of the actual vulnerable lines are in the top 10 predicted lines
            top_10_predictions = tp_predictions[:10]
            top_3_predictions = tp_predictions[:3]
            top_5_predictions = tp_predictions[:5]
            top_1_predictions = tp_predictions[:1]
            
            is_any_vulnerable_in_top_10 = any(pred in ground_truth for pred in top_10_predictions)
            if is_any_vulnerable_in_top_10:
                top_10_accuracies.append(100)
            else:
                top_10_accuracies.append(0)
                
            is_any_vulnerable_in_top_3 = any(pred in ground_truth for pred in top_3_predictions)
            if is_any_vulnerable_in_top_3:
                top_3_accuracies.append(100)
            else:
                top_3_accuracies.append(0)
                
            is_any_vulnerable_in_top_5 = any(pred in ground_truth for pred in top_5_predictions)
            if is_any_vulnerable_in_top_5:
                top_5_accuracies.append(100)
            else:
                top_5_accuracies.append(0)
                
            is_any_vulnerable_in_top_1 = any(pred in ground_truth for pred in top_1_predictions)
            if is_any_vulnerable_in_top_1:
                top_1_accuracies.append(100)
            else:
                top_1_accuracies.append(0)

            # Calculate Initial False Alarm (IFA)
            initial_false_alarm = None
            for idx, pred in enumerate(tp_predictions):
                if pred in ground_truth:
                    initial_false_alarm = idx
                    break
            
            if initial_false_alarm is not None:
                total_min_clean_lines_inspected += initial_false_alarm
            elif len(tp_predictions) != 0:
                total_min_clean_lines_inspected += len(tp_predictions)
            else:
                total_min_clean_lines_inspected += loc

            # Calculate Effort@20%Recall and Recall@1%LOC
            actual_vuln_lines = set(ground_truth)
            predicted_lines = tp_predictions
            
            # Calculate effort until 20% of vulnerable lines are found
            if correctly_identified_vuln_lines < twenty_percent_vuln_lines:
                for line in predicted_lines:
                    effort_to_find_vuln += 1
                    if line in actual_vuln_lines:
                        correctly_identified_vuln_lines += 1
                    if correctly_identified_vuln_lines == twenty_percent_vuln_lines:
                        break
            
            # Calculate Recall@1%LOC
            top_one_percent_predicted = set(predicted_lines[:one_percent_loc])
            vuln_lines_in_top_one_percent += len(top_one_percent_predicted.intersection(actual_vuln_lines))
            
            # print("predicted_lines", predicted_lines)
            # print("Total lines", total_lines)
            # print("total_vulnerable_lines", total_vulnerable_lines)
            # print("one_percent_loc", one_percent_loc, total_lines * 0.01)
            # print("top_one_percent_predicted", top_one_percent_predicted)
            # print("vuln_lines_in_top_one_percent", vuln_lines_in_top_one_percent)
            # print("effort_to_find_vuln", effort_to_find_vuln)
        
    effort_at_20_recall = effort_to_find_vuln / total_lines if total_lines else 0
    recall_at_one_percent_loc = vuln_lines_in_top_one_percent / total_vulnerable_lines if total_vulnerable_lines else 0

    #print("total_min_clean_lines_inspected", total_min_clean_lines_inspected)
    #print("total_functions", total_functions)

    if total_functions == 0:
        overall_top_10_accuracy = 0
        overall_top_3_accuracy = 0
        overall_top_5_accuracy = 0
        overall_top_1_accuracy = 0
        ifa = 100
    else:
        # top_10_accuracy
        hit = sum([1 for acc in top_10_accuracies if acc == 100])
        overall_top_10_accuracy = hit / total_functions
        
        hit = sum([1 for acc in top_3_accuracies if acc == 100])
        overall_top_3_accuracy = hit / total_functions
        
        hit = sum([1 for acc in top_5_accuracies if acc == 100])
        overall_top_5_accuracy = hit / total_functions
        
        hit = sum([1 for acc in top_1_accuracies if acc == 100])
        overall_top_1_accuracy = hit / total_functions
        
        # IFA
        ifa = round(total_min_clean_lines_inspected / total_functions, 2) if total_functions > 0 else 0
        
    return overall_top_1_accuracy, overall_top_3_accuracy, overall_top_5_accuracy, overall_top_10_accuracy, ifa, effort_at_20_recall, recall_at_one_percent_loc
        
def avg_experiments(file_name, language, print_data=False, number_of_experiments = 10, skip_id=[], overlap_lang_experiment=None, total_lines_dict=None, total_vulnerable_lines_dict=None, line_level=True, func_level=True):
    
    if 'linevul' in file_name:
        is_gpt = False
    else:
        is_gpt = True
    
    zero_f1_ids = []
        
    # Initialize variables to accumulate metrics
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_accuracy = 0
    total_mcc = 0
    
    total_top_1_accuracy = 0
    total_top_3_accuracy = 0
    total_top_5_accuracy = 0
    total_top_10_accuracy = 0
    total_ifa = 0
    total_effort_at_20_recall = 0
    total_recall_at_one_percent_loc = 0
    
    for experiment_id in range (number_of_experiments):
        if experiment_id in skip_id:
            continue
        if print_data:
            print(f"Experiment {experiment_id}")
            
        if line_level:
            if overlap_lang_experiment is not None:
                overlap = overlap_lang_experiment[f"{language}_{experiment_id}"]
            else:
                overlap = None
                
            top_1_accuracy, top_3_accuracy, top_5_accuracy, top_10_accuracy, ifa, effort_at_20_recall, recall_at_one_percent_loc = calculate_line_metrics_from_file(
                f"result/{language}/{experiment_id}/{file_name}", 
                is_gpt=is_gpt, 
                print_data=print_data, 
                overlap=overlap,
                total_lines = total_lines_dict[f"{language}_{experiment_id}"],
                total_vulnerable_lines = total_vulnerable_lines_dict[f"{language}_{experiment_id}"])
                
            total_top_10_accuracy += top_10_accuracy
            total_ifa += ifa
            total_top_3_accuracy += top_3_accuracy
            total_top_5_accuracy += top_5_accuracy
            total_top_1_accuracy += top_1_accuracy
            total_effort_at_20_recall += effort_at_20_recall
            total_recall_at_one_percent_loc += recall_at_one_percent_loc
            
        if func_level:
            precision, recall, f1, accuracy, mcc = calculate_func_metrics_from_file(
                f"result/{language}/{experiment_id}/{file_name}", 
                is_gpt=is_gpt, 
                print_data=print_data)
            
            if f1 == 0:
                zero_f1_ids.append(experiment_id)

            total_precision += precision
            total_recall += recall
            total_f1_score += f1
            total_accuracy += accuracy
            total_mcc += mcc
        
    # Calculate number of experiment for line level and func level
    number_of_experiments = number_of_experiments - len(skip_id)
    print(f"Number of iterations {number_of_experiments}")
    
    if line_level:
        avg_top_10_accuracy = total_top_10_accuracy / number_of_experiments
        avg_top_5_accuracy = total_top_5_accuracy / number_of_experiments
        avg_top_3_accuracy = total_top_3_accuracy / number_of_experiments
        avg_top_1_accuracy = total_top_1_accuracy / number_of_experiments
        avg_ifa = total_ifa / number_of_experiments
        avg_effort_at_20_recall = total_effort_at_20_recall / number_of_experiments
        avg_recall_at_one_percent_loc = total_recall_at_one_percent_loc / number_of_experiments
        
        print()
        print(f"\tAverage top 1 accuracy: {round(avg_top_1_accuracy * 100, 2)}%")
        print(f"\tAverage top 3 accuracy: {round(avg_top_3_accuracy * 100, 2)}%")
        print(f"\tAverage top 5 accuracy: {round(avg_top_5_accuracy * 100, 2)}%")
        print(f"\tAverage top 10 accuracy: {round(avg_top_10_accuracy * 100, 2)}%")
        print(f"\tAverage ifa: {round(avg_ifa, 2)}")
        print(f"\tAverage effort at 20 recall: {round(avg_effort_at_20_recall, 4)}")
        print(f"\tAverage recall at 1% loc: {round(avg_recall_at_one_percent_loc, 4)}")
        
        return avg_top_1_accuracy, avg_top_3_accuracy, avg_top_5_accuracy, avg_top_10_accuracy, avg_ifa, avg_effort_at_20_recall, avg_recall_at_one_percent_loc
        
    if func_level:
        avg_precision = total_precision / number_of_experiments
        avg_recall = total_recall / number_of_experiments
        avg_f1_score = total_f1_score / number_of_experiments
        avg_accuracy = total_accuracy / number_of_experiments
        avg_mcc = total_mcc / number_of_experiments
        
        if len(zero_f1_ids) != 0:
            print(f"List zero f1-score experiments: {zero_f1_ids}")
        print()
        print(f"\tAverage Mcc: {round(avg_mcc, 2)}")
        print(f"\tAverage F1 Score: {round(avg_f1_score, 2)}")
        print(f"\tAverage Precision: {round(avg_precision, 2)}")
        print(f"\tAverage Recall: {round(avg_recall, 2)}")
        print(f"\tAverage Accuracy: {round(avg_accuracy * 100, 2)}%")
        
        return avg_mcc, avg_f1_score, avg_precision, avg_recall, avg_accuracy
        
    print()

def get_vulnerable_line_numbers(line_level_predictions: str, source_code: str) -> list:
    """
    Given a string of vulnerable lines and the source code, returns the line numbers
    of vulnerable lines in the source code.
    
    Args:
    - line_level_predictions (str): A string representing the vulnerable lines with potentially incorrect line numbers.
    - source_code (str): The source code as a string.

    Returns:
    - list: List of line numbers representing the vulnerable lines in the source code.
    """
    
    # Split line_level_predictions into lines and extract just the code from each line
    predictions_lines = line_level_predictions.strip().split('\n')
    
    vulnerable_lines_from_predictions = []
    for line in predictions_lines:
        if '`' in line:
            vulnerable_lines_from_predictions.append(line.split('`')[1].strip())
        else:
            # Extracting the line content after the colon (considering no backticks are present)
            try:
                vulnerable_lines_from_predictions.append(line.split(': ', 1)[1].strip())
            except:
                pass
                #print(f"line_level_predictions: {line_level_predictions}")
                #print(f"Source code: {source_code}")
                #print(f"Error line: {line}")

    # Parse source code into lines
    source_code_lines = [line.strip() for line in source_code.split('\n')]
    
    # print("vulnerable_lines_from_predictions", vulnerable_lines_from_predictions)
    # print()
    # print("source code lines:")
    # for line in source_code_lines:
    #     print(line)
    # print()
    
    # Find the line numbers for the lines in line_level_predictions in the source code
    vulnerable_line_numbers = []
    for line in vulnerable_lines_from_predictions:
        for idx, source_line in enumerate(source_code_lines):
            if line == source_line or line in source_line or line.startswith(source_line):
                if idx in vulnerable_line_numbers:
                    continue
                vulnerable_line_numbers.append(idx)
                break  # Break once found, no need to keep checking other source lines for this predicted line
    
    return vulnerable_line_numbers

def get_include_ids(skip_ids):
    if skip_ids is None:
        return list(range(10))
    
    skip_ids = set(skip_ids)
    
    # Define the range of all possible experiment IDs (0 through 9 in this case)
    all_ids = set(range(10))

    # Calculate the IDs to skip
    include_ids = all_ids - skip_ids

    return sorted(list(include_ids))