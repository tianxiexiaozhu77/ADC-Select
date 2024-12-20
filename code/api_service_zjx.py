import json
import requests
import os
from openai import OpenAI
import time
import numpy as np
import time
import vthread
from mistral_inference.generate import generate
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest



def api_get_tokens(api_name,input_text,res_dict,output_fout,logger,args,model,tokenizer):  # args.api_name,input,res_dict,f,logger
    
    client = OpenAI(
            base_url="https://www.gptapi.us/v1",
            # api_key="sk-FOEU2G0AJjUv8ROtA883A69f70A947A8Bf340511D8B0053e"
            # base_url="https://api.pumpkinaigc.online/v1",
            api_key="sk-BzRCbeyJsTdHXlK19c692e0619744aBfBcB8C58fD653Dc7d"  # args.api_key_1106
            # base_url="https://api.pumpkinaigc.online/v1",
            # api_key="sk-UQHnVDbBEbMG40Yd4a373054Ba0640Ba977f1b84B23173F5"
        )
    
    if 'Llama' in args.api_name:
        messages = [
            {"role": "system", "content": "You are llama. You are a helpful assistant."},
            {"role": "user", "content": input_text}]
        text = tokenizer.apply_chat_template(
                                                messages,
                                                tokenize=False,
                                                add_generation_prompt=True
                                            )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    if 'Qwen' in args.api_name:
        messages = [
            {"role": "system", "content": "You are Qwen. You are a helpful assistant."},
            {"role": "user", "content": input_text}]
        text = tokenizer.apply_chat_template(
                                                messages,
                                                tokenize=False,
                                                add_generation_prompt=True
                                            )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
    if 'deepseek' in args.api_name:
        messages = [
            {"role": "system", "content": "You are deepseek. You are a helpful assistant."},
            {"role": "user", "content": input_text}]
        text = tokenizer.apply_chat_template(
                                                messages,
                                                tokenize=False,
                                                add_generation_prompt=True
                                            )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
    if 'Mistral' in args.api_name:
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=input_text)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        
        
    while True:
        try:
            if 'gpt' in args.api_name:
                response = client.chat.completions.create(
                            model=api_name, # api_name,
                            messages=[
                                {
                                    "role": "user",
                                    "content": input_text
                                }
                            ],
                            temperature=0,
                        )
                ans=response.choices[0].message.content.replace("\n","\\n")
                res_dict["pred_answer"]=ans
                output_fout.write(str(res_dict)+"\n")
                output_fout.flush()
                logger.info(api_name)
                break
            if 'Llama' in args.api_name or 'Qwen' in args.api_name or 'deepseek' in args.api_name:
                generated_ids = model.generate(
                                **model_inputs,
                                max_new_tokens=2048,
                                pad_token_id=tokenizer.eos_token_id,
                                # top_p=0.9,
                                temperature=0.001,  # 0相当于将所有概率质量转移到最可能的令牌。
                                do_sample=False,
                            )
                generated_ids = [
                                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                ans = response
                res_dict["pred_answer"]=ans
                output_fout.write(str(res_dict)+"\n")
                output_fout.flush()
                logger.info(api_name)
                break
            if 'Mistral' in args.api_name:
                out_tokens, _ = generate([tokens], model, max_tokens=2048, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
                response = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
                ans = response
                res_dict["pred_answer"]=ans
                output_fout.write(str(res_dict)+"\n")
                output_fout.flush()
                logger.info(api_name)
                break

                

        except Exception as e:
            if logger is not None:
                logger.info(e)
                logger.info(str(res_dict["idx"])+" "+input_text)
            else:
                print(e)
            time.sleep(1)
            continue
        
        
        
def llama_get_tokens(api_name,input_text,res_dict,output_fout,logger,args,model,tokenizer):
    
    messages = [
            {"role": "system", "content": "You are llama. You are a helpful assistant."},
            {"role": "user", "content": input_text}]
    text = tokenizer.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True
                                        )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # client = OpenAI(
    #         # base_url="https://www.gptapi.us/v1",
    #         # api_key="sk-BzRCbeyJsTdHXlK19c692e0619744aBfBcB8C58fD653Dc7d"
    #         base_url="https://api.pumpkinaigc.online/v1",
    #         api_key=args.api_key_1106
    #     )

    while True:
        try:
            # response = client.chat.completions.create(
            #             model=api_name,
            #             messages=[
            #                 {
            #                     "role": "user",
            #                     "content": input_text
            #                 }
            #             ],
            #             temperature=0,
            #         )
            # ans=response.choices[0].message.content.replace("\n","\\n")
            generated_ids = model.generate(
                                **model_inputs,
                                max_new_tokens=2048,
                                pad_token_id=tokenizer.eos_token_id
                            )
            generated_ids = [
                                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            ans = response
            res_dict["pred_answer"]=ans
            output_fout.write(str(res_dict)+"\n")
            output_fout.flush()
            break

        except Exception as e:
            if logger is not None:
                logger.info(e)
                logger.info(str(res_dict["idx"])+" "+input_text)
            else:
                print(e)
            time.sleep(1)
            continue
