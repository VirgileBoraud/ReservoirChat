github_urls = [
    'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/1-Getting_Started.ipynb',
    'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/2-Advanced_Features.ipynb',
    'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/3-General_Introduction_to_Reservoir_Computing.ipynb',
    'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/4-Understand_and_optimize_hyperparameters.ipynb',
    'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/5-Classification-with-RC.ipynb',
    'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/6-Interfacing_with_scikit-learn.ipynb',
]

notebook_paths = ['doc/notebook/notebook_{}.ipynb'.format(i) for i in range(1, len(github_urls)+1)]
print(notebook_paths)