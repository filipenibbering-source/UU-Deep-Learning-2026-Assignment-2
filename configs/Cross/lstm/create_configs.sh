hidden_sizes=(8 32 128)
num_layers=(1)
bidirectionals=(true false)

for hidden_size in "${hidden_sizes[@]}"; do
    for bidirectional in "${bidirectionals[@]}"; do
        for num_layer in "${num_layers[@]}"; do
cat > "lstm_cross_${hidden_size}_${bidirectional}_${num_layer}.yaml" <<EOL

hidden_size: ${hidden_size}
bidirectional: ${bidirectional}
num_layers: ${num_layer}
data_type: cross
EOL

        done
    done
done