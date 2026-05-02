See prompts/completed.md for what this codebase was previously doing. However, I discovered a fatal issue in the codebase: the static inference code doesn't check if there's a valid norm stats path for a given dataset. If the path is unresolved, it defaults to None and takes unnormalized data.

I want you to impose a minor modification on the code so that it prints the norm stats path that the model is supposed to load from in the .out file, and if that file is not found, the code simply exists and errors norm stats path not found at _path.

The static inference code is in static_inference/ Its launcher is also there. Try to make the modification as simple as possible while achieving the goal (must exist if the norm stats file is invalid)


Tell me where decides where the model is drawing its current norm stats paths from