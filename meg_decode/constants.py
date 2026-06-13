"""Shared label definitions for the MEG task."""

CLASS_NAMES = ("rest", "math_story", "working_memory", "motor")

TASK_TO_CLASS = {
    "rest": "rest",
    "task_story_math": "math_story",
    "task_working_memory": "working_memory",
    "task_motor": "motor",
}

CLASS_TO_LABEL = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
LABEL_TO_CLASS = {index: class_name for class_name, index in CLASS_TO_LABEL.items()}
TASK_TO_LABEL = {
    task_name: CLASS_TO_LABEL[class_name] for task_name, class_name in TASK_TO_CLASS.items()
}

