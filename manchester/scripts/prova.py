import pathlib
path1=str(pathlib.Path("mean_car_count_per_grid.csv").parent.resolve())

path1= path1.replace("scripts","csm")+"nome.smc"
print("L'indirizzo è \n")
print(path1)
