from openai import OpenAI
import json

client = OpenAI(api_key="")

with open("/Users/srijonsarkar/Downloads/ad-hoc/adverse_drug_rx/cadac_instruction_test.jsonl", "r") as f:
    for i, row in enumerate(f):
        data = json.loads(row.strip())
        
        completion = client.chat.completions.create(
            # model="ft:gpt-4o-mini-2024-07-18:personal:cadec-dataset:BkPRKdCH",
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": data["messages"][0]["content"]},
                {"role": "user", "content": data["messages"][1]["content"]}
            ]
        )

        response = completion.choices[0].message.content
        print(i)
        print(response)
        print(data["messages"][2]["content"])

        # out_data = {
        #     "messages": data["messages"],
        #     "openai_4o_mini": response
        # }

        data["openai_4o_mini"] = response
        with open("/Users/srijonsarkar/Downloads/ad-hoc/adverse_drug_rx/cadac_instruction_test_zero_shot_gpt_outputs.jsonl", "a") as f2:
            json.dump(data, f2)
            f2.write("\n")

