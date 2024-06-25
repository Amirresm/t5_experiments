from human_eval.data import write_jsonl, read_problems

problems = read_problems()

# num_samples_per_task = 200
# samples = [
#     dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
#     for task_id in problems
#     for _ in range(num_samples_per_task)
# ]
# write_jsonl("samples.jsonl", samples)

problem_list = []
for task_id in problems:
    problem_list.append(problems[task_id])

print(problem_list[4].keys())
print(problem_list[4])
