import openai
import os


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)

model_name = "gpt-3.5-turbo"
jsonl_file = "ft_agent.jsonl"
file_id = "your_file_id"
job_id = "your_job_id"

# File upload


def file_upload(jsonl_file):
    with open(jsonl_file, "rb") as file:
        response = openai.File.create(
            file=file,
            purpose='fine-tune'
        )
    file_id = response['id']
    print(f"File uploaded successfully with ID: {file_id}")
    return file_id


# Job upload
def job_upload(file_id):
    response = openai.FineTuningJob.create(
        training_file=file_id,
        model=model_name,
        #        hyperparameters={"n_epochs": value,},
    )
    job_id = response['id']
    print(f"Fine-tuning job created successfully with ID: {job_id}")
    return job_id


# Retrieve_job
def retrieve_job(job_id):
    response = openai.FineTuningJob.list_events(id=job_id, limit=50)
    events = response['data']
    events.reverse()
    for event in events:
        print(event["message"])


# file_id = file_upload(jsonl_file)
# job_id = job_upload(file_id)
retrieve_job(job_id)

print(openai.FineTune.retrieve(job_id))
