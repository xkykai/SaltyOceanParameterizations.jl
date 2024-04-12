FILE_DIR = "./batch_scripts/NDE_enzyme_localbaseclosure_convectivetanh_shearlinear_uvTSdrhodz_20sim" 
mkpath(FILE_DIR)

hidden_layers = [1, 2, 3]
hidden_layer_sizes = [128, 256, 512]
activations = ["relu", "swish"]

for hidden_layer in hidden_layers, hidden_layer_size in hidden_layer_sizes, activation in activations
    filename = "$(hidden_layer)l_ls$(hidden_layer_size)_$(activation).sh"
    filepath = "$(FILE_DIR)/$(filename)"
    touch(filepath)

    file = open(filepath, "w")

    write(file, "#!/bin/bash\n",
                "source /etc/profile\n",
                "cd ~/SaltyOceanParameterizations.jl\n",
                "ulimit -s unlimited\n",
                "export JULIA_NUM_THREADS=2\n",
                "unbuffer ~/julia-1.10.1/bin/julia --project train_NDE_enzyme_localbaseclosure_convectivetanh_shearlinear_TSrho_args.jl --hidden_layer $(hidden_layer) --hidden_layer_size $(hidden_layer_size) --activation $(activation)")
    close(file)
end

hidden_layers = [1, 2]
hidden_layer_sizes = [1024]
activations = ["relu", "swish"]

for hidden_layer in hidden_layers, hidden_layer_size in hidden_layer_sizes, activation in activations
    filename = "$(hidden_layer)l_ls$(hidden_layer_size)_$(activation).sh"
    filepath = "$(FILE_DIR)/$(filename)"
    touch(filepath)

    file = open(filepath, "w")

    write(file, "#!/bin/bash\n",
                "source /etc/profile\n",
                "cd ~/SaltyOceanParameterizations.jl\n",
                "ulimit -s unlimited\n",
                "export JULIA_NUM_THREADS=2\n",
                "unbuffer ~/julia-1.10.1/bin/julia --project train_NDE_enzyme_localbaseclosure_convectivetanh_shearlinear_TSrho_args.jl --hidden_layer $(hidden_layer) --hidden_layer_size $(hidden_layer_size) --activation $(activation)")
    close(file)
end