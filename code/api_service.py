import json
import requests
import os
from openai import OpenAI
import time
import numpy as np
import time
import vthread

def api_get_tokens(api_name,input_text,res_dict,output_fout,logger,model,tokenizer):


    res_dict["pred_answer"]=None
    output_fout.write(str(res_dict)+"\n")
    output_fout.flush()
    

    # client = OpenAI(
    #         base_url="https://api.pumpkinaigc.online/v1",
    #         api_key="********"  # args.api_key_1106
    #     )

    # while True:
    #     try:
    #         response = client.chat.completions.create(
    #                     model=api_name,
    #                     messages=[
    #                         {
    #                             "role": "user",
    #                             "content": input_text
    #                         }
    #                     ],
    #                     temperature=0,
    #                 )
    #         ans=response.choices[0].message.content.replace("\n","\\n")
    #         res_dict["pred_answer"]=ans
    #         output_fout.write(str(res_dict)+"\n")
    #         output_fout.flush()
    #         break

    #     except Exception as e:
    #         if logger is not None:
    #             logger.info(e)
    #             logger.info(str(res_dict["idx"])+" "+input_text)
    #         else:
    #             print(e)
    #         time.sleep(1)
    #         continue
