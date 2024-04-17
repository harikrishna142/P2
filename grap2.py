import networkx as nx
import json
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv

with open('clea_com3.json', 'r',encoding='utf-8') as file1:
    data = json.load(file1)
p=set()
for k in data:
    p.add(k["author"])

p=list(p)

# Assuming 'sample_data' is your loaded JSON data containing posts and comments
G = nx.Graph()

# Helper function to preprocess text for node attributes (simple version)
for post in data:
    post_id = post["id"]
    author = post["author"]
    body = post["body"]
    
    idea = post["idea"]
    # Add post node with author as an attribute. Now, every node represents a post or a comment-as-post.
    G.add_node(post_id, author=author,body=body,idea=idea)
    
    # Connect this post to other posts by the same author
    for other_node in G:
        if G.nodes[other_node].get('author') == author and other_node != post_id:
            G.add_edge(post_id, other_node)
    
    # Process comments as if they were posts by the comment author
    for comment in post["comments"]:
        comment_id = comment["id"]  # Treat the comment ID as a unique identifier, similar to a post ID
        comment_author = comment["comment_author"]
        c_body = comment["body"]
    
        c_idea = "Indicator"
        # Ensure the comment author and post author are not the same, no self-loops
        if author != comment_author and comment_author in p:
            # Add the comment as a node, treating it as a post by the comment author
            G.add_node(comment_id, author=comment_author,body=c_body,idea=c_idea)
            
            # Connect the comment-post to the original post
            G.add_edge(comment_id, post_id)
            
            # Connect the comment-post to other posts by the same comment author
            for other_node in G:
                if G.nodes[other_node].get('author') == comment_author and other_node != comment_id:
                    G.add_edge(comment_id, other_node)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Assuming your NetworkX graph `G` has nodes with 'body' attribute
texts = [G.nodes[node]['body'] for node in G.nodes]
# Convert text to embeddings using SBERT
text_embeddings = model.encode(texts)

# Function to encode categorical labels (for 'symptoms' or 'idea')
def encode_labels(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)

# Example using 'symptoms' as the label to encode
symptoms_labels = [G.nodes[node]['idea'] for node in G.nodes]
symptoms_encoded = encode_labels(symptoms_labels)

# Convert edge list to edge_index tensor
edge_list = list(G.edges())
#edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
node_mapping = {node: i for i, node in enumerate(G.nodes())}
numeric_edge_list = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in G.edges()]

# Now, convert your numeric edge list to a tensor
edge_index = torch.tensor(numeric_edge_list, dtype=torch.long).t().contiguous()
"""class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, 8, heads=8, dropout=0.6)  # First GAT layer
        # Output of the first layer would be 8 (features per head) * 8 (heads) = 64
        self.conv2 = GATConv(8 * 8, output_dim, heads=1, concat=False, dropout=0.6)  # Second GAT layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
"""
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, 16, normalize=True, aggr='mean')  # First GraphSAGE layer
        self.conv2 = SAGEConv(16, output_dim, normalize=True, aggr='mean')  # Second GraphSAGE layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create PyTorch Geometric data object
data = Data(x=torch.tensor(text_embeddings, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(symptoms_encoded, dtype=torch.long))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(input_dim=data.x.size(1), output_dim=4).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(data)
# After constructing the graph with features
n1 = nx.number_connected_components(G)
#nx.write_gexf(G, "graph_with_features2.gexf")
num_nodes = data.num_nodes
num_training_nodes = int(num_nodes * 0.8)

# Create masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Randomly select indices for training and testing
indices = torch.randperm(num_nodes)
train_indices = indices[:num_training_nodes]
test_indices = indices[num_training_nodes:]

# Assign to masks
train_mask[train_indices] = True
test_mask[test_indices] = True

# Update the data object
data.train_mask = train_mask
data.test_mask = test_mask
# Assuming all other setup (including GCN class definition) is done as previously shown


# Training loop (now with train_mask properly defined)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # Use train_mask for loss calculation
    loss.backward()
    optimizer.step()

# Prediction and evaluation (now with test_mask properly defined)
model.eval()
_, pred = model(data).max(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
total = data.test_mask.sum().item()
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

#nx.write_gexf(G, "gr5.gexf")
