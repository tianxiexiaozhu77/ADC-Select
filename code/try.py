from openai import OpenAI

client = OpenAI(
        base_url="https://www.gptapi.us/v1",
        api_key="sk-fvB1sPI0PdSkgVCVC81870F8C35b46A0A029090f7036E2Df"
    )

question = 'Did Barack Obama add Women\'s rowing to the Olympic programme for the first time as a president?'
# question = '2015 is coming in 36 hours. What is the date a month ago in MM/DD/YYYY? Options: (A) 03/07/2015 (B) 11/28/2014 (C) 11/23/2014 (D) 12/20/2014 (E) 12/05/2014 (F) 11/29/2014'
# question = "2015 is coming in 36 hours. What is the date a month ago in MM/DD/YYYY?\nOptions:\n(A) 03/07/2015\n(B) 11/28/2014\n(C) 11/23/2014\n(D) 12/20/2014\n(E) 12/05/2014\n(F) 11/29/2014"

prompt = str(
    f"Given question: {question}" 
    f"please output 1 demonstrations with similar sentence structures and semantics. "
    "You must only output in a parsible JSON format. Two example outputs look like: \n"
    "Example 1: {{\"question\": \"Demonstration similar to the given question.\"}}\n"
    "Example 2: {{\"question\": \"Demonstration similar to the given question.\"}}\n"
    "Output: "
    )

# prompt = str(
#     f"Given sentence: {question} \n" 
#     f"please output 5 demonstrations with similar sentence structures and semantics. "
#     "You must only output in a parsible JSON format. Two example outputs look like: \n"
#     "Example 1: {{\"question\": \"Demonstration similar to the given sentence.\"}}\n"
#     "Example 2: {{\"question\": \"Demonstration similar to the given sentence.\"}}\n"
#     "Output: "
#     )

# 'gpt-4o'
# 'gpt-3.5-turbo-1106'
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.7,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5
)

cleaned_res = response.choices[0].message.content.replace('```json\n', '').replace('\n```', '').replace('\n',' ')
res = eval(cleaned_res)
print(res)
# import pdb; pdb.set_trace()