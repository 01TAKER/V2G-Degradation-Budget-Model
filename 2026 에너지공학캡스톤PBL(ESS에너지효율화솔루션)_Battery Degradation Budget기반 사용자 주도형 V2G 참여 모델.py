import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. 전기차(EV) 객체 클래스 정의 (물리 모델 및 예산 제약 조건 포함)
# =================================================================
class ElectricVehicle:
    def __init__(self, name, capacity_kwh, soh_pct, budget_pct, current_soc):
        self.name = name                      # 차량 이름 (예: BMW i7)
        self.capacity_kwh = capacity_kwh      # 배터리 정격 용량 (kWh)
        self.soh_pct = soh_pct                # 현재 배터리 건강 상태 (SOH %)
        
        # 사용자가 설정한 '열화 예산(Budget)' (0.1% 등)
        self.budget_pct = budget_pct / 100.0
        self.current_soc = current_soc / 100.0 # 현재 배터리 잔량 (SOC)
        
        self.degradation_accumulated = 0.0     # 시뮬레이션 중 누적된 SOH 열화량 추적
        self.total_energy_supplied_kwh = 0.0   # 전력망에 공급한 누적 에너지량
        self.is_active = True                  # 현재 스케줄링 참여 가능 여부 
        
        # 제약 조건
        # 1. 차주가 퇴근할 때 차가 방전되어 있으면 안 되므로 최소 20% 잔량 보장
        self.min_soc = 0.20
        # 2. 직류(배터리) -> 교류(건물) 변환 시 발생하는 인버터 효율 손실 (92% 가정)
        self.inverter_efficiency = 0.92

    # 물리적 열화량 계산
    def calculate_degradation(self, c_rate, dt_hours, temperature):
        # 최적 온도(25도)에서 벗어날수록 열화가 가속되는 온도 페널티 적용
        temp_penalty = 1.0 + abs(temperature - 25) * 0.02
        
        # CPfade 특성 반영: SOH가 낮을수록(오래된 배터리일수록) 추가 열화 속도가 둔화
        soh_factor = self.soh_pct / 100.0
        # C-rate가 커질수록 열화가 비선형적으로 증가함을 지수(1.5)로 표현
        deg_step = (c_rate ** 1.5) * dt_hours * 0.001 * soh_factor * temp_penalty
        return deg_step

    # 실제 방전 수행 및 예산 삭감 로직
    def discharge(self, target_kw, dt_hours, temperature):
        if not self.is_active or self.current_soc <= self.min_soc:
            return 0.0
            
        c_rate = target_kw / self.capacity_kwh
        if c_rate > 1.0: c_rate = 1.0
        
        required_battery_kw = (c_rate * self.capacity_kwh) / self.inverter_efficiency
        energy_to_discharge = required_battery_kw * dt_hours
        soc_drop = energy_to_discharge / self.capacity_kwh
        
        # 제약 1: 배터리 하한선(20%)
        if self.current_soc - soc_drop < self.min_soc:
            energy_to_discharge = (self.current_soc - self.min_soc) * self.capacity_kwh
            self.current_soc = self.min_soc
            self.is_active = False 
            required_battery_kw = energy_to_discharge / dt_hours
            c_rate = (required_battery_kw * self.inverter_efficiency) / self.capacity_kwh
            
        deg_step = self.calculate_degradation(c_rate, dt_hours, temperature)
        
        # 제약 2: 사용자가 허용한 열화 예산(Budget) 한도 체크
        if self.degradation_accumulated + deg_step >= self.budget_pct:
            remaining_deg = self.budget_pct - self.degradation_accumulated
            actual_kw = required_battery_kw * (remaining_deg / deg_step) * self.inverter_efficiency
            self.degradation_accumulated = self.budget_pct
            self.is_active = False # 영구 차단
        else:
            self.degradation_accumulated += deg_step
            self.current_soc -= soc_drop
            actual_kw = required_battery_kw * self.inverter_efficiency
            
        energy_supplied = actual_kw * dt_hours
        self.total_energy_supplied_kwh += energy_supplied
        
        return energy_supplied

# =================================================================
# 2. V2B 스케줄링 시뮬레이션 메인 엔진 (개별 로그 추적 통합)
# =================================================================
def run_simulation(fleet, load_profile, pv_profile, temp_profile, dt_hours, v2g_price, grid_price):
    original_eb = []
    shaved_eb = []
    enterprise_savings = 0
    total_compensation = 0
    
    ev_logs = {ev.name: [] for ev in fleet}

    for load, pv, temp in zip(load_profile, pv_profile, temp_profile):
        eb_t = load - pv 
        original_eb.append(max(eb_t, 0))
        
        if eb_t > 0:
            active_evs = [ev for ev in fleet if ev.is_active]
            supplied_t = 0
            
            if active_evs:
                target_kw = eb_t / len(active_evs)
                for ev in fleet:
                    if ev in active_evs:
                        discharged = ev.discharge(target_kw, dt_hours, temp)
                        ev_logs[ev.name].append(discharged)
                        supplied_t += discharged
                    else:
                        ev_logs[ev.name].append(0.0)
            else:
                for ev in fleet:
                    ev_logs[ev.name].append(0.0)
                    
            shaved_load = eb_t - supplied_t
            shaved_eb.append(max(shaved_load, 0))
            enterprise_savings += supplied_t * grid_price
            total_compensation += supplied_t * v2g_price
        else:
            shaved_eb.append(0)
            for ev in fleet:
                ev_logs[ev.name].append(0.0)

    return original_eb, shaved_eb, enterprise_savings, total_compensation, ev_logs

# =================================================================
# 3. 데이터 초기화 및 시나리오 실행
# =================================================================
np.random.seed(42)
time_steps = np.arange(9, 19)
temperature_data = [22, 24, 26, 28, 30, 31, 30, 28, 25, 23]

V2G_COMP_PRICE = 190      
GRID_PEAK_PRICE = 350     
dt_hours = 1.0           

base_load_6 = np.array([50, 70, 100, 150, 180, 160, 120, 90, 70, 60])
base_pv_6   = np.array([10, 30,  50,  60,  50,  40,  20, 10,  0,  0])
actual_load_6 = base_load_6 * np.random.normal(1.0, 0.05, len(base_load_6))
actual_pv_6   = base_pv_6   * np.random.normal(1.0, 0.05, len(base_pv_6))

fleet_6 = [
    ElectricVehicle("BMW i7 (New)",    101.5, 100, 0.05, 90),
    ElectricVehicle("BMW i7 (Used)",   101.5,  85, 0.05, 90),
    ElectricVehicle("Kia EV9",          96.0, 100, 0.10, 80),
    ElectricVehicle("Tesla Model S",    95.0,  85, 0.15, 75),
    ElectricVehicle("BYD Seal",         82.5, 100, 0.10, 85),
    ElectricVehicle("Nissan Leaf",      40.0,  85, 0.20, 90),
]

orig_eb_6, shaved_eb_6, savings_6, comp_6, ev_logs_6 = run_simulation(
    fleet_6, actual_load_6, actual_pv_6, temperature_data, dt_hours, V2G_COMP_PRICE, GRID_PEAK_PRICE
)

# --- 100대 생략 부분 ---
base_load_100 = base_load_6 * 10
base_pv_100   = base_pv_6   * 10
actual_load_100 = base_load_100 * np.random.normal(1.0, 0.05, len(base_load_100))
actual_pv_100   = base_pv_100   * np.random.normal(1.0, 0.05, len(base_pv_100))

fleet_100 = []
car_capacities = [40.0, 58.0, 77.4, 82.5, 95.0, 101.5]
for i in range(100):
    cap    = np.random.choice(car_capacities)
    soh    = np.random.uniform(80, 100)
    budget = np.random.uniform(0.05, 0.25)
    soc    = np.random.uniform(60, 100)
    fleet_100.append(ElectricVehicle(f"EV_{i+1}", cap, soh, budget, soc))

orig_eb_100, shaved_eb_100, savings_100, comp_100, ev_logs_100 = run_simulation(
    fleet_100, actual_load_100, actual_pv_100, temperature_data, dt_hours, V2G_COMP_PRICE, GRID_PEAK_PRICE
)

# =================================================================
# 4. 시각화 1 (기존 2x3 통합 그래프)
# =================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

axes[0, 0].plot(time_steps, orig_eb_6, 'r--', label='Original Grid Load', linewidth=2)
axes[0, 0].plot(time_steps, shaved_eb_6, 'b', label='Load after V2B', linewidth=2)
axes[0, 0].fill_between(time_steps, orig_eb_6, shaved_eb_6, color='green', alpha=0.2, label='Peak Shaved')
axes[0, 0].set_title('[6 EVs] Peak Shaving', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Time (Hour)', fontsize=12)
axes[0, 0].set_ylabel('Energy Deficit (kW)', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True)

categories_6 = ['Grid Savings', 'Compensation', 'Infra Cost', 'Net Profit']
infra_cost_6 = 15000
net_profit_6 = savings_6 - comp_6 - infra_cost_6
values_6 = [savings_6, -comp_6, -infra_cost_6, net_profit_6]
colors_6 = ['green', 'red', 'red', 'blue']
axes[0, 1].bar(categories_6, values_6, color=colors_6)
axes[0, 1].axhline(0, color='black', linewidth=1)
axes[0, 1].set_title('[6 EVs] Daily ROI Analysis', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Amount (KRW)', fontsize=12)
for i, v in enumerate(values_6):
    axes[0, 1].text(i, v + (3000 if v > 0 else -6000), f'{int(v):,} KRW', ha='center', fontweight='bold')

sohs_6 = [ev.soh_pct for ev in fleet_6]
supplied_6 = [ev.total_energy_supplied_kwh for ev in fleet_6]
names_6 = [ev.name for ev in fleet_6]
axes[0, 2].scatter(sohs_6, supplied_6, color='purple', s=120)
for i, name in enumerate(names_6):
    axes[0, 2].annotate(name, (sohs_6[i], supplied_6[i]), xytext=(7, 7), textcoords='offset points')
axes[0, 2].set_title('[6 EVs] Initial SOH vs Supplied Energy', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Initial SOH (%)', fontsize=12)
axes[0, 2].set_ylabel('Supplied Energy (kWh)', fontsize=12)
axes[0, 2].grid(True)

axes[1, 0].plot(time_steps, orig_eb_100, 'r--', label='Original Grid Load', linewidth=2)
axes[1, 0].plot(time_steps, shaved_eb_100, 'b-', label='Load after V2B (VPP)', linewidth=2)
axes[1, 0].fill_between(time_steps, orig_eb_100, shaved_eb_100, color='green', alpha=0.3, label='Peak Shaved (VPP)')
axes[1, 0].set_title('[100 EVs] Peak Shaving (VPP Scale)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Time (Hour)', fontsize=12)
axes[1, 0].set_ylabel('Energy Deficit (kW)', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True)

categories_100 = ['Grid Savings', 'Compensation', 'Infra Cost', 'Net Profit']
infra_cost_100 = 250000
net_profit_100 = savings_100 - comp_100 - infra_cost_100
values_100 = [savings_100, -comp_100, -infra_cost_100, net_profit_100]
colors_100 = ['green', 'red', 'red', 'blue']
axes[1, 1].bar(categories_100, values_100, color=colors_100)
axes[1, 1].axhline(0, color='black', linewidth=1)
axes[1, 1].set_title('[100 EVs] Daily ROI Analysis (VPP Scale)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Amount (KRW)', fontsize=12)
for i, v in enumerate(values_100):
    axes[1, 1].text(i, v + (30000 if v > 0 else -60000), f'{int(v):,} KRW', ha='center', fontweight='bold')

sohs_100 = [ev.soh_pct for ev in fleet_100]
supplied_100 = [ev.total_energy_supplied_kwh for ev in fleet_100]
budgets_100 = [ev.budget_pct * 100 for ev in fleet_100]
scatter = axes[1, 2].scatter(sohs_100, supplied_100, c=budgets_100, cmap='viridis', s=60, alpha=0.8, edgecolors='w')
cbar = plt.colorbar(scatter, ax=axes[1, 2])
cbar.set_label('User Budget (%)', fontsize=10)
axes[1, 2].set_title('[100 EVs] SOH vs Supplied Energy (By Budget)', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Initial SOH (%)', fontsize=12)
axes[1, 2].set_ylabel('Supplied Energy (kWh)', fontsize=12)
axes[1, 2].grid(True)

# =================================================================
# 5. 시각화 2: 개별 차량 방전 로그 (마커 겹침 방지 X축 이동 적용)
# =================================================================
fig2, ax = plt.subplots(figsize=(10, 6))

explicit_colors = ['blue', 'darkorange', 'green', 'red', 'purple', 'saddlebrown']
markers = ['o', 's', '^', 'D', 'v', 'p']
linestyles = ['-', '--', '-.', ':', '-', '--']

for idx, (name, logs) in enumerate(ev_logs_6.items()):
    x_offset = (idx - 2.5) * 0.04 

    ax.plot(time_steps + x_offset, logs, label=name, color=explicit_colors[idx], 
            linestyle=linestyles[idx], marker=markers[idx], 
            linewidth=2, markersize=8, alpha=0.85)

ax.set_title('[6 EVs] Individual Discharge Profile (Budget & SOC Constraint)', fontsize=16, fontweight='bold')
ax.set_xlabel('Time (Hour)', fontsize=14)
ax.set_ylabel('Discharged Energy (kWh)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.7)

# X축 표시를 원래 시간 단위 정수로 고정
ax.set_xticks(time_steps) 

# Nissan Leaf 차단 포인트 (SOC 부족)
ax.annotate('Nissan Leaf\nBlocked (SOC < 20%)', 
            xy=(13, 0.2), xytext=(9.5, 3),
            arrowprops=dict(facecolor='saddlebrown', shrink=0.05, width=2, headwidth=8),
            fontsize=12, color='saddlebrown', fontweight='bold')

# BMW i7 신차 차단 포인트 (예산 소진)
ax.annotate('BMW i7 (New)\nBudget Exhausted', 
            xy=(14, 0.2), xytext=(11.5, 6),
            arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8),
            fontsize=12, color='blue', fontweight='bold')

# BMW i7 중고차 생존 포인트 (예산 소진 지연)
ax.annotate('BMW i7 (Used)\nSurvived Longer', 
            xy=(15, 0.2), xytext=(14.5, 5),
            arrowprops=dict(facecolor='darkorange', shrink=0.05, width=2, headwidth=8),
            fontsize=12, color='darkorange', fontweight='bold')

plt.show()
