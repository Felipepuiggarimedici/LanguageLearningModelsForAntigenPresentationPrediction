import os

nHeads = [30]
nLayers = [30]
d_ff = [512]
d_k = [64]

for head in nHeads:
    for layer in nLayers:
        for feedForw in d_ff:
            for dimInHeads in d_k:
                command = (
                    f'qsub -v nHeads={head},nLayers={layer},d_ff={feedForw},d_k={dimInHeads} run5Fold.pbs'
                )
                print("Submitting:", command)
                os.system(command)
