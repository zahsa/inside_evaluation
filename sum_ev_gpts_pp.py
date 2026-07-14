import openai
# from openai import OpenAI
import json
import argparse
import tqdm
import time
import pandas as pd
if __name__ == '__main__':


    tempt = 1
    n_completions = 20
    gpt_name = 'gpt-4o-mini' # gpt-4.1-mini, gpt-5-mini, gpt-5-mini-2025-08-07
    dim = 'RE'
    param = f"sum_{dim}_{gpt_name}_temp{tempt}_n{n_completions}"
    input_prompt = 'rel_detailed.txt'
    
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--prompt_fp', type=str, default='/users/all_data/prompts/summeval/' + input_prompt)

    argparser.add_argument('--save_fp', type=str, default='/users/all_data/results/sum/' + param + '.json') 
    argparser.add_argument('--summeval_fp', type=str, default='/users/faculty/all_data/data/summeval.json')

    argparser.add_argument('--model', type=str, default= gpt_name) #'gpt-4.1-nano,gpt-4-0613'
    # argparser.add_argument('--key', type=str, required=True)

    
    args = argparser.parse_args()
    
       
    argskey = 'your_app_key' 

    openai.api_key = argskey

    summeval = json.load(open(args.summeval_fp))
    prompt = open(args.prompt_fp).read()

    ct, ignore = 0, 0

    new_json = []
    for instance in tqdm.tqdm(summeval):
        source = instance['source']
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
        instance['prompt'] = cur_prompt
        while True:
            try:
                _response = openai.ChatCompletion.create(
                # _response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": cur_prompt}],
                    temperature=tempt, 
                    max_completion_tokens=5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    # logprobs=40,
                    n=n_completions  
                )
                time.sleep(0.5)
                # print('evaluated')
                # print('response',_response)

                all_responses = [_response['choices'][i]['message']['content'] for i in
                                 range(len(_response['choices']))]
                instance['all_responses'] = all_responses
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(f"Error: {e}")  # Print the general exception message
                print("Detailed Error:")
                traceback.print_exc()  # Print the full traceback for debugging
        

                
                if ("limit" in str(e)):
                    time.sleep(2)
                else:
                    ignore += 1
                    print('ignored', ignore)
                    # instance['all_responses'] = 20*[-1]
                    # new_json.append(instance)
                    break

    print('ignored total', ignore)
    print('----json',new_json)
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
