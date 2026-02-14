import torch
import pandas as pd
import numpy as np
import joblib
from models.informer import Informer

config = {
    'seq_len': 24,
    'label_len': 6,
    'pred_len': 1,
    'enc_in': 25,       # number of input features (excluding target)
    'dec_in': 25,
    'c_out': 1,
    'd_model': 256,
    'n_heads': 8,
    'e_layers': 3,
    'd_layers': 2,
    'd_ff': 1024,
    'dropout': 0.1,
    'attn': 'prob',
    'embed': 'timeF',
    'freq': 'm',
    'activation': 'gelu',
    'output_attention': False,
    'distil': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

device = config['device']

model = Informer(
    enc_in=config['enc_in'], dec_in=config['dec_in'], c_out=config['c_out'],
    seq_len=config['seq_len'], label_len=config['label_len'], pred_len=config['pred_len'],
    factor=5, d_model=config['d_model'], n_heads=config['n_heads'],
    e_layers=config['e_layers'], d_layers=config['d_layers'], d_ff=config['d_ff'],
    dropout=config['dropout'], attn=config['attn'], embed=config['embed'], freq=config['freq'],
    activation=config['activation'], output_attention=config['output_attention'], distil=config['distil']
).to(device)

model.load_state_dict(
    torch.load("./checkpoints/Ganga-BOD-Test/Ganga-BOD-Test.pth", map_location=device)
)
model.eval()


scaler_input = joblib.load("scaler_input.pkl")  
column_order = joblib.load("column_order.pkl")

# Ensure BOD is not used as input
if "BOD (mg/l)" in column_order:
    column_order = [c for c in column_order if c != "BOD (mg/l)"]


#LOAD DATA
data = pd.read_csv("put dataset url")
data.columns = data.columns.str.strip()

data['Month'] = pd.to_datetime(data['Month'], format='%m')
data['month'] = data['Month'].dt.month - 1
data.drop(columns=['Month'], inplace=True)

base_year = data['Year'].min()
data['year'] = data['Year'] - base_year
data.drop(columns=['Year'], inplace=True)


location_stats = {}
for loc in data['Location'].unique():
    loc_bod = data[data['Location'] == loc]['BOD (mg/l)'].values.astype(np.float32)
    location_stats[int(loc)] = {
        'mean': float(loc_bod.mean()),
        'std': float(loc_bod.std()) if float(loc_bod.std()) > 0 else 1e-6
    }

#    PREDICTION FUNCTION
def predict_bod(location_num: int, target_year: int, target_month: int):   
    target_year_offset = target_year - base_year   
    df = data[data['Location'] == location_num].sort_values(['year', 'month'])
    if len(df) < config['seq_len']:
        return None  
    
    seq_df = df.iloc[-config['seq_len']:].copy()
    
    X = seq_df.drop(columns=['BOD (mg/l)'])
    X = X[column_order]  
    
    X_scaled = scaler_input.transform(X)
    seq_x = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    enc_mark_np = seq_df[['month', 'year']].values.astype(np.float32)
    enc_mark = torch.tensor(enc_mark_np, dtype=torch.float32).unsqueeze(0).to(device)

    dec_inp = torch.zeros(
        (1, config['label_len'] + config['pred_len'], config['dec_in']),
        dtype=torch.float32
    ).to(device)
    
    future_mark = np.array([[target_month - 1, target_year_offset]], dtype=np.float32)
    dec_mark_np = np.concatenate([enc_mark_np[-config['label_len']:], future_mark], axis=0)
    dec_mark = torch.tensor(dec_mark_np, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(seq_x, enc_mark, dec_inp, dec_mark)

    pred_scaled = float(out[:, -1, 0].item())
    
    if location_num not in location_stats:
        raise ValueError(f"Unknown location: {location_num}")

    loc_mean = location_stats[location_num]['mean']
    loc_std = location_stats[location_num]['std']
    bod_pred = pred_scaled * loc_std + loc_mean

    return float(bod_pred)

