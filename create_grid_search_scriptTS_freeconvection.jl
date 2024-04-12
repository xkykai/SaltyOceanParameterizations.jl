FILE_DIR = "./batch_scripts/NDE_enzyme_NN_localbaseclosure_convectivetanh_shearlinear_TSrho_TSdrhodz_FC" 
mkpath(FILE_DIR)

hidden_layers = [1, 2]
hidden_layer_sizes = [256, 512]
S_scalings = [0.1, 1., 10, 100]

for hidden_layer_size in hidden_layer_sizes, hidden_layer in hidden_layers, S_scaling in S_scalings
    filename = "layersize_$(hidden_layer_size)_$(activation).sh"
    filepath = "$(FILE_DIR)/$(filename)"
    touch(filepath)

    file = open(filepath, "w")

    write(file, "#!/bin/bash\n",
                "source /etc/profile\n",
                "cd ~/SaltyOceanParameterizations.jl\n",
                "ulimit -s unlimited\n",
                "export JULIA_NUM_THREADS=2\n",
                "unbuffer ~/julia-1.10.1/bin/julia --project train_NDE_enzyme_NN_localbaseclosure_convectivetanh_shearlinear_freeconvection_TSrho_TSdrhodz_weighting.jl --hidden_layer_size $(hidden_layer_size) --hidden layer $(hidden_layer) --S_scaling $(S_scaling) --activation swish")
    close(file)
end