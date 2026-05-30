import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from graph import build_graph
from routing import load_dataset, time_to_interval, predict_flow, find_top_k, format_time


#Load everything once at startup
print("loading graph and dataset...")
nodes, edges = build_graph()
df = load_dataset()
traffic_cols = [f"V{str(i).zfill(2)}" for i in range(96)]
print("ready")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("TBRGS - Traffic Based Route Guidance System")
        self.root.geometry("1200x750")
        self.root.configure(bg="#1e1e2e")

        self.current_routes = []
        self.selected_route = 0

        self.build_ui()


    def build_ui(self):
        #Left panel - inputs and results
        left = tk.Frame(self.root, bg="#1e1e2e", width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left.pack_propagate(False)

        title = tk.Label(left, text="TBRGS", font=("Courier", 22, "bold"), bg="#1e1e2e", fg="#cdd6f4")
        title.pack(pady=(10, 2))

        subtitle = tk.Label(left, text="Traffic Based Route Guidance", font=("Courier", 9), bg="#1e1e2e", fg="#6c7086")
        subtitle.pack(pady=(0, 15))

        #Input fields
        self.origin_var = tk.StringVar(value="2000")
        self.dest_var = tk.StringVar(value="3002")
        self.time_var = tk.StringVar(value="08:00")
        self.model_var = tk.StringVar(value="lstm")
        self.algo_var = tk.StringVar(value="astar")

        fields = [
            ("Origin SCATS site", self.origin_var),
            ("Destination SCATS site", self.dest_var),
            ("Time (HH:MM)", self.time_var),
        ]

        for label, var in fields:
            tk.Label(left, text=label, font=("Courier", 9), bg="#1e1e2e", fg="#a6adc8").pack(anchor="w", pady=(8,1))
            tk.Entry(left, textvariable=var, font=("Courier", 11), bg="#313244", fg="#cdd6f4",
                     insertbackground="#cdd6f4", relief="flat", bd=5).pack(fill=tk.X)

        tk.Label(left, text="Model", font=("Courier", 9), bg="#1e1e2e", fg="#a6adc8").pack(anchor="w", pady=(8,1))
        model_frame = tk.Frame(left, bg="#1e1e2e")
        model_frame.pack(fill=tk.X)
        for m in ["lstm", "gru", "cnn_lstm"]:
            tk.Radiobutton(model_frame, text=m.upper(), variable=self.model_var, value=m,
                           font=("Courier", 9), bg="#1e1e2e", fg="#cdd6f4",
                           selectcolor="#313244", activebackground="#1e1e2e").pack(side=tk.LEFT)

        tk.Label(left, text="Algorithm", font=("Courier", 9), bg="#1e1e2e", fg="#a6adc8").pack(anchor="w", pady=(8,1))
        algo_frame = tk.Frame(left, bg="#1e1e2e")
        algo_frame.pack(fill=tk.X)
        for a in [("A*", "astar"), ("UCS", "ucs")]:
            tk.Radiobutton(algo_frame, text=a[0], variable=self.algo_var, value=a[1],
                           font=("Courier", 9), bg="#1e1e2e", fg="#cdd6f4",
                           selectcolor="#313244", activebackground="#1e1e2e").pack(side=tk.LEFT)

        tk.Button(left, text="Find Routes", command=self.run_search,
                  font=("Courier", 11, "bold"), bg="#89b4fa", fg="#1e1e2e",
                  relief="flat", bd=0, pady=8, cursor="hand2").pack(fill=tk.X, pady=15)

        #Results list
        tk.Label(left, text="Routes", font=("Courier", 10, "bold"), bg="#1e1e2e", fg="#cdd6f4").pack(anchor="w")

        self.results_frame = tk.Frame(left, bg="#1e1e2e")
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(left, text="", font=("Courier", 9), bg="#1e1e2e", fg="#6c7086", wraplength=290)
        self.status_label.pack(pady=5)


        #Right panel - map and chart
        right = tk.Frame(self.root, bg="#181825")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0,10), pady=10)

        self.fig, (self.ax_map, self.ax_chart) = plt.subplots(2, 1, figsize=(7, 8))
        self.fig.patch.set_facecolor("#181825")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.draw_map()
        self.draw_empty_chart()
        self.canvas.draw()


    #Draw the base network map
    def draw_map(self, highlight_path=None):
        self.ax_map.clear()
        self.ax_map.set_facecolor("#181825")

        lats = [nodes[n]["lat"] for n in nodes]
        lons = [nodes[n]["lon"] for n in nodes]

        highlight_edges = set()
        if highlight_path and len(highlight_path) > 1:
            for i in range(len(highlight_path) - 1):
                highlight_edges.add((highlight_path[i], highlight_path[i+1]))
                highlight_edges.add((highlight_path[i+1], highlight_path[i]))

        # draw edges
        drawn = set()
        for site, neighbours in edges.items():
            for n, d in neighbours:
                if (site, n) not in drawn:
                    drawn.add((site, n))
                    drawn.add((n, site))
                    if site not in nodes or n not in nodes:
                        continue
                    x = [nodes[site]["lon"], nodes[n]["lon"]]
                    y = [nodes[site]["lat"], nodes[n]["lat"]]
                    if (site, n) in highlight_edges:
                        self.ax_map.plot(x, y, color="#89b4fa", linewidth=2.5, zorder=2)
                    else:
                        self.ax_map.plot(x, y, color="#313244", linewidth=1, zorder=1)

        # draw nodes
        for site in nodes:
            lat = nodes[site]["lat"]
            lon = nodes[site]["lon"]
            if highlight_path and site in highlight_path:
                color = "#a6e3a1" if site == highlight_path[-1] else "#89b4fa"
                if site == highlight_path[0]:
                    color = "#f38ba8"
                self.ax_map.scatter(lon, lat, color=color, s=60, zorder=4)
                self.ax_map.annotate(site, (lon, lat), fontsize=6, color="#cdd6f4",
                                     xytext=(3, 3), textcoords="offset points", zorder=5)
            else:
                self.ax_map.scatter(lon, lat, color="#45475a", s=25, zorder=3)

        self.ax_map.set_title("Boroondara Network", color="#cdd6f4", fontsize=10, pad=8)
        self.ax_map.tick_params(colors="#45475a", labelsize=7)
        for spine in self.ax_map.spines.values():
            spine.set_edgecolor("#313244")


    #Draw empty traffic chart placeholder
    def draw_empty_chart(self):
        self.ax_chart.clear()
        self.ax_chart.set_facecolor("#181825")
        self.ax_chart.set_title("Predicted Traffic (select a route)", color="#6c7086", fontsize=10)
        self.ax_chart.tick_params(colors="#45475a")
        for spine in self.ax_chart.spines.values():
            spine.set_edgecolor("#313244")


    #Draw 24hr traffic prediction for origin site
    def draw_traffic_chart(self, site_id, model_name):
        self.ax_chart.clear()
        self.ax_chart.set_facecolor("#181825")

        group = df[df["SCATS Number"] == site_id]
        raw = group[traffic_cols].astype(float).values.flatten()
        series = pd.Series(raw).ffill().bfill().values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

        predictions = []
        for i in range(96):
            flow = predict_flow(site_id, i, df, model_name=model_name)
            predictions.append(flow)

        hours = [i * 15 / 60 for i in range(96)]

        # actual average
        avg_actual = []
        for i in range(96):
            vals = group[traffic_cols[i]].astype(float).dropna()
            avg_actual.append(vals.mean() * 4)

        self.ax_chart.plot(hours, avg_actual, color="#45475a", linewidth=1, label="actual avg", linestyle="--")
        self.ax_chart.plot(hours, predictions, color="#89b4fa", linewidth=1.5, label=f"{model_name} predicted")

        self.ax_chart.set_title(f"Traffic prediction - site {site_id}", color="#cdd6f4", fontsize=10)
        self.ax_chart.set_xlabel("hour of day", color="#6c7086", fontsize=8)
        self.ax_chart.set_ylabel("veh/hr", color="#6c7086", fontsize=8)
        self.ax_chart.tick_params(colors="#6c7086", labelsize=7)
        self.ax_chart.legend(fontsize=7, facecolor="#313244", labelcolor="#cdd6f4")
        for spine in self.ax_chart.spines.values():
            spine.set_edgecolor("#313244")


    #Run the route search
    def run_search(self):
        origin = self.origin_var.get().strip().zfill(4)
        dest = self.dest_var.get().strip().zfill(4)
        time_str = self.time_var.get().strip()
        model = self.model_var.get()
        algo = self.algo_var.get()

        if origin not in nodes:
            messagebox.showerror("Error", f"Site {origin} not found in graph")
            return
        if dest not in nodes:
            messagebox.showerror("Error", f"Site {dest} not found in graph")
            return

        self.status_label.config(text="searching...")
        self.root.update()

        try:
            interval = time_to_interval(time_str)
            routes = find_top_k(nodes, edges, origin, dest, interval, df, k=5, algorithm=algo, model_name=model)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="")
            return

        self.current_routes = routes
        self.status_label.config(text=f"found {len(routes)} routes")

        # clear old results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        for i, (path, cost, created) in enumerate(routes):
            self.add_route_button(i, path, cost, created, model)

        if len(routes) > 0:
            self.select_route(0, routes[0][0], model)


    def add_route_button(self, index, path, cost, created, model):
        frame = tk.Frame(self.results_frame, bg="#313244", cursor="hand2")
        frame.pack(fill=tk.X, pady=2)

        label = tk.Label(frame,
            text=f"Route {index+1}  —  {format_time(cost)}\n{' → '.join(path)}",
            font=("Courier", 8), bg="#313244", fg="#cdd6f4",
            justify="left", wraplength=280, pady=5, padx=8)
        label.pack(anchor="w")

        nodes_label = tk.Label(frame, text=f"nodes created: {created}",
            font=("Courier", 7), bg="#313244", fg="#6c7086", pady=2, padx=8)
        nodes_label.pack(anchor="w")

        frame.bind("<Button-1>", lambda e, i=index, p=path, m=model: self.select_route(i, p, m))
        label.bind("<Button-1>", lambda e, i=index, p=path, m=model: self.select_route(i, p, m))
        nodes_label.bind("<Button-1>", lambda e, i=index, p=path, m=model: self.select_route(i, p, m))


    def select_route(self, index, path, model):
        self.selected_route = index
        self.draw_map(highlight_path=path)
        self.draw_traffic_chart(path[0], model)
        self.fig.tight_layout(pad=2)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()