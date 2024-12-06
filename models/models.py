import pandas as pd
from openai import AsyncOpenAI, AsyncAzureOpenAI
import os
import asyncio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncAzureOpenAI(
    azure_endpoint="https://notes-app-resource.openai.azure.com/",
    api_version="2024-08-01-preview",
    api_key=OPENAI_API_KEY,
)
MODEL = 'gpt-4o-mini'

class SimpleEventModel:
    def __init__(self, dataframe: pd.DataFrame, event_name: str):
        self.dataframe = dataframe
        self.event_keywords = event_name.split()

    def predict_attendance(self) -> pd.DataFrame:
        def contains_keywords(text: str) -> int:
            if pd.isna(text):
                return 0
            return int(all(keyword.lower() in text.lower() for keyword in self.event_keywords))

        self.dataframe['BASELINE_PREDICTED_ATTENDANCE'] = self.dataframe['POST_TEXT'].apply(contains_keywords)
        return self.dataframe

##baseline prompt:
PROMPT1 = """Here is a post please read it very carefully:

<post>
{post_text}
</post>

<instructions>
You are a text classification model tasked with identifying whether a LinkedIn post suggests attendance at the event '{event_name}'. 
The goal is to determine if the post suggests attendance (loosely defined as planning, promoting, or showing interest). 
Does this post suggest the user is attending the event? Respond with ONLY 1 for attendance or 0 otherwise. Your output should be ONLY 0/1.
</instructions>
"""


### prompt with more context:
PROMPT2 = """Here is a post please read it very carefully:

<post>
{post_text}
</post>

<context>
Post date: {post_date}
Event date: {event_date}
Location: Cologne
Country: Germany
</context>

<instructions>
You are a text classification model tasked with identifying whether a LinkedIn post suggests attendance at the event '{event_name}'. 
The goal is to determine if the post suggests attendance (loosely defined as planning, promoting, or showing interest). IMPORTANT: Keep in mind that the attendance has to be for same YEAR and location as the event and NOT a previous year.
Does this post suggest the user is attending the event? Respond with ONLY 1 for attendance or 0 otherwise. Your output should be ONLY 0/1.
</instructions>
"""

class PromptBasedEventModel:
    default_prompt = PROMPT2

    def __init__(self, dataframe: pd.DataFrame, event_name: str, event_date: pd.Timestamp, model: str = MODEL, prompt: str = None):
        self.dataframe = dataframe
        self.event_name = event_name
        self.event_date = event_date
        self.model = model
        self.prompt = prompt or self.default_prompt

    async def classify_post(self, row) -> int:
        post_text = row['POST_TEXT']
        post_date = row['DATE_PUBLISHED']
        
        if not post_text or pd.isna(post_text):
            return 0

        prompt = self.prompt.format(
            post_text=post_text,
            post_date=post_date.strftime('%Y-%m-%d'),
            event_date=self.event_date.strftime('%Y-%m-%d'),
            event_name=self.event_name
        )

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
            )
            answer = response.choices[0].message.content
            # print(answer)
            return int(answer)
        except Exception as e:
            print(f"ERROR: {e}")
            return 0

    async def predict_attendance(self) -> pd.DataFrame:
        tasks = [self.classify_post(row) for _, row in self.dataframe.iterrows()]
        predictions = await asyncio.gather(*tasks)
        self.dataframe['PROMPT_PREDICTED_ATTENDANCE'] = predictions
        return self.dataframe

def get_metrics(y_true, y_pred):


    conf_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    tp = conf_matrix.iloc[1, 1]
    tn = conf_matrix.iloc[0, 0]
    fp = conf_matrix.iloc[0, 1]
    fn = conf_matrix.iloc[1, 0]

    precision = tp / (tp + fp) 
    recall = tp / (tp + fn) 
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


if __name__ == "__main__":

    file_path = 'attendance_data.xlsx'
    data = pd.ExcelFile(file_path)
    post_details = data.parse('post_details')
    ##this is the test dataset, prompts were not tuned on this dataset:
    post_details = post_details.sample(n=250, random_state=42) 
    post_details['DATE_PUBLISHED'] = pd.to_datetime(post_details['DATE_PUBLISHED'])
    post_details['IS_ATTENDING_NUMERIC'] = post_details['IS_ATTENDING'].map({'yes': 1, 'no': 0})

    # Data is now ready to use
    print(post_details.head())

    event_details = data.parse('event_details')
    event_date = pd.to_datetime(event_details.loc[0, 'start_date'])
    event_name = event_details["event_name"].iloc[0] #"DMEXCO 2024"
    event_city = event_details["city"]
    event_country = event_details["country"]


    # Baseline Model:
    baseline_model = SimpleEventModel(post_details, event_name)
    post_details = baseline_model.predict_attendance()

    # Prompt Model:
    prompt_model1 = PromptBasedEventModel(post_details, event_name, event_date, prompt=PROMPT1)
    post_details = asyncio.run(prompt_model1.predict_attendance())
    post_details = post_details.rename(columns={'PROMPT_PREDICTED_ATTENDANCE': 'PROMPT_PREDICTED_ATTENDANCE_BASE'})

    # Prompt Model2:
    prompt_model2 = PromptBasedEventModel(post_details, event_name, event_date, prompt=PROMPT2)
    post_details = asyncio.run(prompt_model2.predict_attendance())

    y_true = post_details['IS_ATTENDING_NUMERIC']

    # Calculate metrics for both models:
    for model in ['BASELINE_PREDICTED_ATTENDANCE', 'PROMPT_PREDICTED_ATTENDANCE_BASE', 'PROMPT_PREDICTED_ATTENDANCE']:
        print(f"\nMetrics for {model}:")
        post_details.to_csv(f"{model.lower()}.csv", index=False)
        y_pred = post_details[model]
        get_metrics(y_true, y_pred)

