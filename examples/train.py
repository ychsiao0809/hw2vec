import os, sys
from time import strftime
import logging
from pathlib import Path

sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.config import Config
from hw2vec.hw2graph import *
from hw2vec.graph2vec.models import *

cfg = Config(sys.argv[1:])

# path = Path('/app/hw2vec/assets/MY_TJ_RTL_DATA/AES-T1000/src/TjFree')
# path = Path('/app/hw2vec/assets/MY_TJ_RTL_DATA/memctrl-T100/rtl/verilog')
# path = Path('/app/hw2vec/assets/MY_TJ_RTL_DATA/memctrl-T100/rtl/verilog')

# hw2graph = HW2GRAPH(cfg)

# for i in range(1500, 2101, 100):
#     path = Path('/app/hw2vec/assets/MY_TJ_RTL_DATA/AES-T'+str(i)+'-TJ')
#     hw_graph = hw2graph.code2graph(path)
# path = Path('/app/hw2vec/assets/TJ-RTL-toy/TjIn/RS232-T1000')
# path = Path('/app/hw2vec/assets/TJ-RTL-toy/TjFree/det_1011')

# exit()

# Reduce design reliability
# Leak information
# Denial of Service

# hw2graph = HW2GRAPH(cfg)
# hw_graph = hw2graph.code2graph(Path('/app/assets/TJ-RTL-toy-trigger-effects/Free/apb_gpio'))
# print('run gpio')
# exit()
    

''' prepare graph data '''
data_pkl_path = os.path.join(cfg.raw_dataset_path, 'DFG-TJ-RTL.pkl')
# if not cfg.data_pkl_path.exists():


if os.path.exists(data_pkl_path) and cfg.load_pkl_file is True:
    print("load pickle")
    ''' reading graph data from cache '''
    data_proc = DataProcessor(cfg)
    # data_proc.read_graph_data_from_cache(cfg.data_pkl_path)
    data_proc.read_graph_data_from_cache(data_pkl_path)
else:
    ''' converting graph using hw2graph '''
    hw2graph = HW2GRAPH(cfg)
    nx_graphs = []
    for hw_project_path in hw2graph.find_hw_project_folders():
        print(hw_project_path)
        hw_graph = hw2graph.code2graph(hw_project_path)
        nx_graphs.append(hw_graph)

    data_proc = DataProcessor(cfg)
    for hw_graph in nx_graphs:
        data_proc.process(hw_graph)
    # data_proc.cache_graph_data(cfg.data_pkl_path)
    data_proc.cache_graph_data(data_pkl_path)

''' logfile '''
output_path = Path(cfg.output_path, strftime("%Y%m%d%H%M%S")).resolve()
logfile = str(output_path/"logfile")
Path(logfile).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=logfile, level=logging.INFO)    

''' prepare dataset '''
all_graphs = data_proc.get_graphs()

TJ_TYPE = { "Free":0 }

# Make sure TjFree is label 0
isfree = False
for i, t in enumerate(set([i.hw_type for i in all_graphs])):
    if t == "Free":
        isfree = True
        continue
    if isfree:
        TJ_TYPE[t] = i
    else:
        TJ_TYPE[t] = i+1


output_dim = len(TJ_TYPE)
logging.info("%d labels : " % len(TJ_TYPE) + ', '.join([k for k, v in TJ_TYPE.items()]))

with open(str(output_path/"class.json"), "w") as f:
    class_json = json.dumps(TJ_TYPE, indent=4)
    f.write(class_json)


for data in all_graphs:
    data.label = TJ_TYPE[data.hw_type]


# train_graphs, test_graphs = data_proc.split_dataset(ratio=cfg.ratio, seed=cfg.seed, dataset=all_graphs)

test_datasets = [
    'det_1011', # Free
    'RC5',  # Free
    'AES-T100-TJ', # LI
    'AES-T500-TJ', # DoS
    'PIC16F84-RS232-T500', # DoS
    'PIC16F84-T400', # DoS
    'RS232-AES-T1800', # DoS
    'RS232-T901', # DoS
    'AES-T400-TJ', # LI
    'AES-T1000-TJ', # LI
    'AES-T1200-TJ', # LI
    'RS232-AES-T1100', # LI
    'RS232-T400' # LI
    ]

train_graphs = []
test_graphs = []
for data in all_graphs:
    if data.hw_name in test_datasets:
        test_graphs.append(data)
    else:
        train_graphs.append(data)

print("Training data: %d, Testing data: %d" % (len(train_graphs), len(test_graphs)))

train_loader = DataLoader(train_graphs, shuffle=True, batch_size=cfg.batch_size)
valid_loader = DataLoader(test_graphs, shuffle=False, batch_size=1)
# print([i.label for i in train_loader])


''' model configuration '''
model = GRAPH2VEC(cfg)
if cfg.load_model:
    model_path = Path(cfg.model_path)
    if model_path.exists():
        model.load_model(str(model_path/"model.cfg"), str(model_path/"model.pth"))
    else:
        print("No such model path.")
else:
    print()
    print("Initializing model...")
    convs = [
        GRAPH_CONV("gcn", data_proc.num_node_labels, cfg.hidden),
        GRAPH_CONV("gcn", cfg.hidden, cfg.hidden)
    ]
    model.set_graph_conv(convs)

    pool = GRAPH_POOL("sagpool", cfg.hidden, cfg.poolratio)
    model.set_graph_pool(pool)

    readout = GRAPH_READOUT("max")
    model.set_graph_readout(readout)

    # output = nn.Linear(cfg.hidden, cfg.embed_dim)
    output = nn.Linear(cfg.hidden, output_dim)
    model.set_output_layer(output)

''' training '''
print("Training...")
model.to(cfg.device)
trainer = GraphTrainer(cfg, dict((v, k) for k, v in TJ_TYPE.items()), class_weights=data_proc.get_class_weights(train_graphs), output_path=output_path)
trainer.build(model)
trainer.train(train_loader, valid_loader)

''' evaluating and inspecting '''
print()
print("Evaluation...")
trainer.evaluate(cfg.epochs, train_loader, valid_loader)
vis_loader = DataLoader(all_graphs, shuffle=False, batch_size=1)
trainer.visualize_embeddings(vis_loader, "./")