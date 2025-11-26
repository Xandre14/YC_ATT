#!/usr/bin/env python
# coding: utf-8

# In[440]:


#Retrieving and Cleaning the Data

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import zipfile, io, openpyxl, time

def fetch_rfr_items(RSS_URL):
    r = requests.get(RSS_URL, timeout=10)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title")
        link  = item.findtext("link")
        pub   = item.findtext("pubDate")
        items.append({"title": title, "link": link, "pubDate": pub})
    return items


def fetch_zipfiles(items):


    session = requests.Session()
    session.headers.update({
        "User-Agent": "my-eiopa-downloader (for personal research; contact: r3andalex@gmail.com)"
    })

    zipfiles = []

    for i, item in enumerate(items[:34]):  # last 34 months
        for attempt in range(3):
            resp = session.get(item["link"], timeout=20)
            if resp.status_code == 429:

                time.sleep(5 * (attempt + 1))
                continue

            resp.raise_for_status()
            break
        else:
            raise RuntimeError(f"Failed to download {item['link']} after retries")

        zip_bytes = io.BytesIO(resp.content)
        zipfiles.append(zip_bytes)

        # Pause between successful downloads to avoid Server issues
        time.sleep(2)

    return zipfiles


def Clean(zipfiles):

    Dataframes = []

    xlsx_name = None

    with zipfile.ZipFile(zipfiles[0]) as z:

    
        for name in z.namelist():
        
            if "Term_Structures" in name and name.endswith(".xlsx"):
                xlsx_name = name
                break
            
        if xlsx_name is None:
            raise RuntimeError("No *Term_Structures*.xlsx file found in ZIP")
            
        with z.open(xlsx_name) as f:

            xls = pd.ExcelFile(f)

            df = pd.read_excel(xls, sheet_name="RFR_spot_no_VA")

    df.columns = df.iloc[0]

    df = df.iloc[9:,2:]

    df = df.reset_index(drop=True)

    Dataframes.append(df)

    valid_countries = df.columns.tolist()

    for zipbs in zipfiles[1:]:

        xlsx_name = None

        with zipfile.ZipFile(zipbs) as z:

    
            for name in z.namelist():
        
                if "Term_Structures" in name and name.endswith(".xlsx"):
                    xlsx_name = name
                    break
            
            if xlsx_name is None:
                raise RuntimeError("No *Term_Structures*.xlsx file found in ZIP")
            
            with z.open(xlsx_name) as f:

                xls = pd.ExcelFile(f)

                df = pd.read_excel(xls, sheet_name="RFR_spot_no_VA")

        df.columns = df.iloc[0]

        df = df.iloc[9:,2:]

        df = df.reset_index(drop=True)

        df = df.loc[:, df.columns.intersection(valid_countries)]

        df = df.reindex(columns=valid_countries)
        
        df = df.dropna(axis=1, how="any")

        valid_countries = df.columns.tolist()
        

        Dataframes.append(df)

    for i, df in enumerate(Dataframes):

        Dataframes[i] = df.loc[:, df.columns.intersection(valid_countries)]

        Dataframes[i] = df.reindex(columns=valid_countries)

        df_num = Dataframes[i].apply(pd.to_numeric, errors='coerce')

        # Optional: check if NaNs were introduced
        if df_num.isna().any().any():
            print(f"Warning: NaNs found in DataFrame {i}. ")

        Dataframes[i] = df_num.ffill()

    return Dataframes



# In[441]:


RSS_URL = "https://www.eiopa.europa.eu/feed/53/rss_en"
items = fetch_rfr_items(RSS_URL)


# In[443]:


zipfiles = fetch_zipfiles(items)


# In[444]:


Dataframes = Clean(zipfiles)


# In[445]:


#The Model

import torch, math


class YC_ATT(torch.nn.Module):

    def __init__(self, M, L, L_range, y_min, y_max, countries_num, d=8, E=6, p_drop=0.5, alpha=0.25):
        super().__init__()
        
        self.Q = torch.nn.Linear(M,d)
        self.K = torch.nn.Linear(M,d)
        self.V = torch.nn.Linear(M,d)
        self.U = torch.nn.Linear(E, M)
        self.W = torch.nn.Linear(L*d, M)
        self.U_upper = torch.nn.Linear(E, M)
        self.W_upper = torch.nn.Linear(L*d, M)
        self.U_lower = torch.nn.Linear(E, M)
        self.W_lower = torch.nn.Linear(L*d, M)
        self.embedding = torch.nn.Embedding(countries_num, E)

        self.criterion_central = torch.nn.SmoothL1Loss()
        

        self.dropout_att = torch.nn.Dropout(p_drop)

        self.register_buffer("y_min", y_min)
        self.register_buffer("y_max", y_max)
        self.register_buffer("country_indices", torch.arange(countries_num, dtype=torch.long))

        self.M = M
        self.d = d
        self.L = L
        self.L_range = L_range
        self.E = E
        self.alpha = alpha
        

    def forward(self, Y):

        Q = torch.tanh(self.Q(Y))
        K = torch.tanh(self.K(Y))
        V = torch.tanh(self.V(Y))

        X = torch.softmax(Q @ K.transpose(-2,-1) / math.sqrt(self.d), dim=-1) @ V 
        x = X.flatten(-2,-1)

        x = self.dropout_att(x)

        e = self.embedding(self.country_indices)

        a = (self.y_max + self.y_min) / 2
        b = (self.y_max - self.y_min) / 2

        S = torch.asinh( 4 * ( self.W(x) + self.U(e) ) ) / 4
        
        S_lower = torch.exp( self.W_lower(x) + self.U_lower(e)) / 4
        S_upper = torch.exp( self.W_upper(x) + self.U_upper(e)) / 4

        y_hat = a.unsqueeze(-1) + b.unsqueeze(-1)*S
        y_hat_lower = y_hat - S_lower
        y_hat_upper = y_hat + S_upper

        return y_hat, y_hat_lower, y_hat_upper

        

    

    def train_model(self, Y, epochs=100, lr=1e-3, lambda_smooth = 1, device="cpu"):
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        training_batches = (int(self.L_range * 0.8) - self.L)

        validation_batches = (int(self.L_range * 0.9) - self.L)

        best_val_loss = float("inf")
        best_state = None

        def PinballLoss(y_hat, y_true, tau):
            error = (y_true - y_hat)
            return torch.mean(torch.max(tau*error, (tau-1)*error))

        def curvature_penalty(y_hat, y_true):
            r = y_true - y_hat
            diff1 = r[..., 1:] - r[..., :-1]
            diff2 = diff1[..., 1:] - diff1[..., :-1]
            return (diff2 ** 2).mean()

        for epoch in range(epochs):

            self.train()

            train_loss = 0.0
            train_batches = 0

            for batch in range(training_batches):

                Y_batch = Y[:,batch:batch+self.L,:].to(device)

                y_true = Y[:,batch+self.L,:].to(device)

                optimizer.zero_grad()

                y_hat, y_hat_lower, y_hat_upper = model(Y_batch)

                loss = (
                self.criterion_central(y_hat, y_true)
                + PinballLoss(y_hat_lower, y_true, self.alpha)
                + PinballLoss(y_hat_upper, y_true, 1-self.alpha)
                )


                smooth_central = curvature_penalty(y_hat, y_true)
                smooth_lower   = curvature_penalty(y_hat_lower, y_true)
                smooth_upper   = curvature_penalty(y_hat_upper, y_true)

                loss = loss + lambda_smooth * (smooth_central + smooth_lower + smooth_upper)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                if epoch == 0 and batch == 0:
                    print("central_loss:", self.criterion_central(y_hat,y_true), "smooth_pen:", smooth_central)

            
            self.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
 

                for batch in range(training_batches, validation_batches):
                    Y_batch = Y[:, batch:batch+self.L, :].to(device)
                    y_true  = Y[:, batch+self.L,   :].to(device)

                    y_hat, y_hat_lower, y_hat_upper = self(Y_batch)

                    loss = (
                        self.criterion_central(y_hat, y_true)
                        + PinballLoss(y_hat_lower, y_true, self.alpha)
                        + PinballLoss(y_hat_upper, y_true, 1-self.alpha)
                    )

                    smooth_central = curvature_penalty(y_hat, y_true)
                    smooth_lower   = curvature_penalty(y_hat_lower, y_true)
                    smooth_upper   = curvature_penalty(y_hat_upper, y_true)

                    loss = loss + lambda_smooth * (smooth_central + smooth_lower + smooth_upper)

                    val_loss += loss.item()
                    val_batches += 1

            val_loss = val_loss / max(val_batches, 1) if val_batches > 0 else float("nan")

            # track best model 
            if val_batches > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            avg_train_loss = train_loss / max(train_batches, 1)

            print(f"Epoch {epoch+1}/{epochs}  "f"(avg_train_loss)={avg_train_loss:.6f}  val_loss={val_loss:.6f}")
            
        if best_state is not None:
            self.load_state_dict(best_state)


    def test_model(self, Y, device="cpu"):
        self.to(device)
        self.eval()

        validation_batches = (int(self.L_range * 0.9) - self.L)
        testing_batches = self.L_range - self.L

        test_central = 0.0
        test_lower   = 0.0
        test_upper   = 0.0
        test_batches = 0

        def PinballLoss(y_hat, y_true, tau):
            error = (y_true - y_hat)
            return torch.mean(torch.max(tau*error, (tau-1)*error))

        with torch.no_grad():

            for batch in range(validation_batches, testing_batches):

                Y_batch = Y[:, batch:batch+self.L, :].to(device)
                y_true  = Y[:, batch+self.L,   :].to(device)

                y_hat, y_hat_lower, y_hat_upper = self(Y_batch)

                central_loss = self.criterion_central(y_hat, y_true)
                lower_loss   = PinballLoss(y_hat_lower, y_true, self.alpha)
                upper_loss   = PinballLoss(y_hat_upper, y_true, 1-self.alpha)

                test_central += central_loss.item()
                test_lower   += lower_loss.item()
                test_upper   += upper_loss.item()
                test_batches += 1
            

        if test_batches == 0:
            return float("nan"), float("nan"), float("nan")

        test_central /= test_batches
        test_lower   /= test_batches
        test_upper   /= test_batches

        print(
            f"Test central MAE={test_central:.6f}, "
            f"lower pinball={test_lower:.6f}, "
            f"upper pinball={test_upper:.6f}"
        )

        return test_central, test_lower, test_upper

        





# In[446]:


countries = Dataframes[0].columns.tolist()

maturities = Dataframes[0].index.to_numpy()


countries_to_id = {c: i for i, c in enumerate(countries)}

tensors = [torch.tensor(df.values, dtype=torch.float32) for df in Dataframes]

Y = torch.stack(tensors, dim=0)

Y = Y.permute(2,0,1)

countries_num, L_range, M = Y.shape[0], Y.shape[1], Y.shape[2]

L = 4

y_min = Y.amin(dim=(1, 2))
y_max = Y.amax(dim=(1, 2))

model = YC_ATT(
    M=M,                 # number of maturities/features
    L=L,                 # window size
    L_range=L_range,     # time dimension
    y_min=y_min,
    y_max=y_max,
    countries_num=countries_num,
    d=8,                 # latent dimension
    E=6,                 # embedding dimension (must match during __init__)
    p_drop=0.2,
    alpha=0.05
)


# In[462]:


model.train_model(Y, epochs=200, lambda_smooth = 10)


# In[459]:


model.test_model(Y)


# In[460]:


import matplotlib.pyplot as plt

def test_plots(model, Y, maturities, L, L_range, device="cpu"):

    model.to(device)
    model.eval()

    validation_batches = int(L_range * 0.9)   - L
    testing_batches = L_range - L

    with torch.no_grad():
        for batch in range(validation_batches, testing_batches):

            Y_batch = Y[:, batch:batch+L, :].to(device)
            y_true  = Y[:, batch+L,   :].to(device)

            y_hat, y_hat_lower, y_hat_upper = model(Y_batch)

            y_true_np  = y_true.cpu().numpy()
            y_hat_np   = y_hat.cpu().numpy()
            y_lo_np    = y_hat_lower.cpu().numpy()
            y_hi_np    = y_hat_upper.cpu().numpy()

            for c in range(40):

                plt.figure()
                plt.plot(maturities, y_true_np[c,:],  label="True", color="red")
                plt.plot(maturities, y_hat_np[c,:],   label="Central", color="blue")
                plt.fill_between(maturities, y_lo_np[c,:], y_hi_np[c,:], alpha=0.2, color="lightskyblue")

                title_country = countries[c] if c < len(countries) else f"Country {c}"
                plt.title(f"Test step {batch+L+1} – {title_country}")
                plt.xlabel("Maturity")
                plt.ylabel("Yield")
                plt.legend()
                plt.tight_layout()
                plt.show()


# In[463]:


test_plots(model, Y, maturities=maturities, L=L, L_range=L_range)

