LOG_DIR="/home/trevor/coding/rnn_research/spectral-lstm/results";
BS=64;
HX=32;
EPC=10;
LAYERS=1;
DEFAULT_ARGS="--epochs ${EPC} --batch-size ${BS} --layers ${LAYERS}";

declare -a tasks=("add" "mul" "xor") # "bball" "seqmnist")

for i in 1 2 3 4 5
do
    for t in "${tasks[@]}"
    do

        EXP_PATH="${LOG_DIR}/slstm_epoch${EPC}_bs${BS}_hx${HX}_task${t}/seed${i}/";
        echo $EXP_PATH;
        mkdir -p $EXP_PATH
        python3 main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --hx ${HX} --task ${t} --model-type slstm &

        EXP_PATH="${LOG_DIR}/svdlstm_epoch${EPC}_bs${BS}_hx${HX}_task${t}/seed${i}/";
        echo $EXP_PATH;
        mkdir -p $EXP_PATH
        python3 main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --hx ${HX} --task ${t} --model-type svdlstm &
        wait;

    done
done
