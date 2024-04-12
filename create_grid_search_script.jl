FILE_DIR = "./batch_scripts/NDE_enzyme_NN_localbaseclosure_convectivetanh_shearlinear_TSrho_FC_profilegradientinput" 
mkpath(FILE_DIR)

# loss_types = ["mse", "mae", "mme"]
# train_withgradient = ["true", "false"]
# hidden_layer_sizes = [128, 256, 512]
# activations = ["relu", "leakyrelu", "swish"]

# for loss_type in loss_types, train_with_gradient in train_withgradient, hidden_layer_size in hidden_layer_sizes, activation in activations
#     filename = "$(loss_type)_$(train_with_gradient)_$(hidden_layer_size)_$(activation).sh"
#     filepath = "$(FILE_DIR)/$(filename)"
#     touch(filepath)

#     file = open(filepath, "w")

#     write(file, "#!/bin/bash\n",
#                 "source /etc/profile\n",
#                 "module load cuda/11.8\n\n",
#                 "cd ~/SaltyOceanParameterizations.jl\n",
#                 "export JULIA_NUM_THREADS=2\n",
#                 "unbuffer ~/julia-1.10.1/bin/julia --project train_1NDE_NN_localbaseclosure_convectivetanh_shearlinear_rho_freeconvection_profilegradientinput_args.jl --loss_type $(loss_type) --train_with_gradient $(train_with_gradient) --hidden_layer_size $(hidden_layer_size) --activation $(activation)")
#     close(file)
# end

hidden_layer_sizes = [256, 512]
activations = ["relu", "leakyrelu", "swish"]

for hidden_layer_size in hidden_layer_sizes, activation in activations
    filename = "layersize_$(hidden_layer_size)_$(activation).sh"
    filepath = "$(FILE_DIR)/$(filename)"
    touch(filepath)

    file = open(filepath, "w")

    write(file, "#!/bin/bash\n",
                "source /etc/profile\n",
                "cd ~/SaltyOceanParameterizations.jl\n",
                "export JULIA_NUM_THREADS=2\n",
                "unbuffer ~/julia-1.10.1/bin/julia --project train_NDE_enzyme_NN_localbaseclosure_convectivetanh_shearlinear_freeconvection_TSrho_profilegradientinput_args.jl --hidden_layer_size $(hidden_layer_size) --activation $(activation)")
    close(file)
end