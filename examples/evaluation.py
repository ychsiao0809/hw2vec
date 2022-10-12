import os, sys
import json
sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.config import Config
from hw2vec.hw2graph import *
from hw2vec.graph2vec.models import *

def main():
    cfg = Config(sys.argv[1:])

    hw2graph = HW2GRAPH(cfg)
    if os.path.isfile(cfg.raw_dataset_path):
        hw_graph = hw2graph.code2graph(cfg.raw_dataset_path)
        data_proc = DataProcessor(cfg)
        data_proc.process(hw_graph)
    else:
        ''' converting graph using hw2graph '''
        nx_graphs = []
        for hw_project_path in hw2graph.find_hw_project_folders():
            print(hw_project_path)
            hw_graph = hw2graph.code2graph(hw_project_path)
            nx_graphs.append(hw_graph)

        data_proc = DataProcessor(cfg)
        for hw_graph in nx_graphs:
            data_proc.process(hw_graph)
    # data_proc.cache_graph_data(cfg.data_pkl_path)
        
    ''' prepare dataset '''
    all_graphs = data_proc.get_graphs()
    data_loader = DataLoader(all_graphs, shuffle=False, batch_size=1)

    ''' model configuration '''
    model = GRAPH2VEC(cfg)
    model_path = Path(cfg.model_path)
    if model_path.exists():
        model.load_model(str(model_path/"model.cfg"), str(model_path/"model.pth"))

    ''' load classes'''
    with open(str(model_path/"class.json")) as f:
        TJ_TYPE = json.load(f)

    
    print("%d labels : " % len(TJ_TYPE) + ', '.join([k for k, v in TJ_TYPE.items()]))
    # TJ_TYPE = {
    #         'T_LI': 0,
    #         'T_DoS': 1,
    #         'Ao_LI': 2,
    #         'Free': 3,
    # }

    for data in all_graphs:
        data.label = 0

    outputs_tensor, preds, node_attns = inference(cfg, model, data_loader, dict((v, k) for k, v in TJ_TYPE.items()))
    # for i, data in enumerate(data_loader):
    #     print(labels[int(preds[i])])
    # print(outputs_tensor)
    # print(self.labels[int(test_preds[i])])
    # vis_loader = DataLoader(all_graphs, shuffle=False, batch_size=1)
    # trainer.visualize_embeddings(vis_loader, "./")

def inference(config, model, data_loader, classes):
    labels = []
    outputs = []
    node_attns = []
    folder_names = []
    
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(data_loader):
            data.to(config.device)

            output, attn = model.embed_graph(data.x, data.edge_index, data.batch)
            output = model.mlp(output)
            output = F.log_softmax(output, dim=1)

            outputs.append(output.cpu())
            
            if 'pool_score' in attn:
                node_attn = {}
                node_attn["original_batch"] = data.batch.detach().cpu().numpy().tolist()
                node_attn["pool_perm"] = attn['pool_perm'].detach().cpu().numpy().tolist()
                node_attn["pool_batch"] = attn['batch'].detach().cpu().numpy().tolist()
                node_attn["pool_score"] = attn['pool_score'].detach().cpu().numpy().tolist()
                node_attns.append(node_attn)

            # labels += np.split(data.label.cpu().numpy(), len(data.label.cpu().numpy()))

        # outputs = torch.cat(outputs).reshape(-1,2).detach()
        outputs = torch.cat(outputs).reshape(-1, len(classes)).detach()

        # labels_tensor = torch.LongTensor(labels).detach()
        outputs_tensor = torch.FloatTensor(outputs).detach()
        preds = outputs_tensor.max(1)[1].type(torch.FloatTensor).detach()
        # preds = outputs_tensor.max(1)[1].type_as(labels_tensor).detach()

        print(preds)
        # print(clas)
        for i, data in enumerate(data_loader):
            print(data.hw_name[0], classes[int(preds[i])])

    return outputs_tensor, preds, node_attns

if __name__ == '__main__':
    main()