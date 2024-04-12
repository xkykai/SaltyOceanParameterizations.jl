FILE_DIR = "./batch_scripts/LES_training_scripts_3"
mkpath(FILE_DIR)

conditions = ["--QU -5e-4 --QT 0 --QS 0",
              "--QU -2e-4 --QT 0 --QS 0",

              "--QU 0 --QT 5e-5 --QS 0",
              "--QU 0 --QT 1e-5 --QS 0",

              "--QU 0 --QT 0 --QS -5e-5",
              "--QU 0 --QT 0 --QS -2e-5",

              "--QU -1.5e-4 --QT 4.5e-5 --QS 0",
              "--QU -4e-4 --QT 1.5e-5 --QS 0",

              "--QU -1.5e-4 --QT 0 --QS -4.5e-5",
              "--QU -4e-4 --QT 0 --QS -2.5e-5"]

for condition in conditions
    filename = "3_$(filter(x -> !isspace(x), condition)).sh"
    filepath = "$(FILE_DIR)/$(filename)"
    touch(filepath)

    file = open(filepath, "w")

    write(file, "#!/bin/bash\n",
                "source /etc/profile\n",
                "module load cuda/11.8\n\n",
                "cd ~/SaltyOceanParameterizations.jl\n",
                "export JULIA_NUM_THREADS=2\n",
                "unbuffer ~/julia-1.10.1/bin/julia --project run_LES_linearTS_training.jl $(condition) --T_surface 4.3 --S_surface 33.5 --dSdz -1.4e-3 --dTdz 1.3e-3 --f 1.2e-4 --stop_time 2 --time_interval 10 --field_time_interval 60 --pickup true --advection WENO9nu0 --Lx 512 --Ly 512 --Lz 256 --Nx 256 --Ny 256 --Nz 128")
    close(file)
end