import os
from time import strftime, localtime
import pandas as pd
from coptpy import *
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14,
                     'font.family': 'Times New Roman',
                     'mathtext.fontset': 'cm'})


def output_folder_check(scenario_folder):
    def check_path(path_to_check):
        # Check whether the specified path exists or not
        is_exist = os.path.exists(path_to_check)

        if not is_exist:
            # Create a new directory because it does not exist
            os.makedirs(path_to_check)

        return path_to_check

    def check_run_path(run_path, save_time=True):
        if save_time:
            new_path = run_path + '__' + ''.join(strftime("%Y_%m_%d_%H_%M_%S", localtime()))
            os.makedirs(new_path)
            return new_path
        else:
            # Check whether the specified path exists or not
            is_exist = os.path.exists(run_path)
            if not is_exist:
                os.makedirs(run_path)
            return run_path

    path = check_path('results')
    scenario_path = check_run_path(path + '/' + scenario_folder, save_time=True)
    check_path(scenario_path + '/pictures')

    return scenario_path


class ModelData:
    def __init__(self):
        # initialize data variables
        self.scenario_name = None
        self.model_name = None
        self.result_path = None
        self.db = None

        # Sets
        self.BCQT = None  # blend quality component type
        self.C = None  # component
        self.D = None  # day
        self.H = None
        self.N = None  # time interval
        self.P = None  # product
        self.Q = None  # quality
        self.R = None  # reservoir
        self.T = None  # tank

        # Parameters
        self.CQ = None  # component quality
        self.CQ_init = None  # component initial quality
        self.CQR_init = None  # component storage initial quality in reservoir
        self.C_cost = None  # component cost
        self.C_init = None  # component init stock
        self.C_pump_max = None
        self.C_pump_min = None
        self.C_stock_max = None  # component max stock
        self.CR_stock = None  # component storage value in reservoir
        self.G = None  # product is gasoline
        self.map_CR = None  # reservoir to component mapping
        self.map_PC = None  # component to product map
        self.map_PT = None  # tank to product mapping
        self.ND = None  # day to time interval
        self.PD_plan = None  # product day plan
        self.PN_max = None  # product time interval max plan
        self.PN_min = None  # product time interval min plan
        self.PQT_init = None  # product storage initial quality in tank
        self.PQ_max = None  # product quality max specification
        self.PQ_min = None  # product quality min specification
        self.PT_stock_init = None  # product storage value in tank
        self.P_blend_min = None  # product minimum blend weight
        self.P_cost = None  # product cost
        self.P_init = None  # product init stock
        self.P_loss = None  # product blend loss coefficient
        self.P_passport_time = None  # product quality checking time
        self.P_prepare_time = None  # product quality checking time

        self.QBM = None  # quality blend method
        self.Q_i = None  # component blend quality index
        self.Q_lb = None  # component blend quality linear bonus
        self.Q_tag = None  # quality check necessity
        self.Q_vc = None  # volumetric calculation of quality

        self.R_max = None  # reservoir max capacity
        self.R_min = None  # reservoir min capacity
        self.R_pump_max = None  # component reservoir pump max
        self.R_pump_min = None  # component reservoir pump min
        self.S = None  # component supply

        self.T_dead_stock = None  # dead stock in product tank
        self.T_max = None  # tank max capacity
        self.T_min = None  # tank min capacity
        self.T_pump_max = None  # product tank pump max
        self.T_pump_min = None  # product tank pump min

        self.CQ_f = {}
        self.PQ_max_f = {}
        self.PQ_min_f = {}
        self.B = None
        self.b2d = None
        # self.p_b = {}

        self.rcl_gap = None
        self.big_m = None

    def import_data_excel(self, config, result_path):
        # get data from GDX
        self.H = config['time_horizon']
        self.model_name = config['Model name']
        self.result_path = result_path
        self.db = pd.read_excel('init_data/' + config['input_file'], sheet_name=None)
        self.rcl_gap = 0.2
        self.big_m = 1E5

        self.read_sets()
        self.read_parameters()
        # self.read_scalars()
        self.calc_q()
        self.create_params()

    def create_params(self):
        self.S = {domains: self.S[domains] / 24 for domains in self.S.keys()}
        self.B = [b + 1 for b in range(2 * sum([self.G[val] for val in self.G.keys()]) + 1)]
        self.b2d = {(b, d): 1 for b in self.B for d in self.D}

    # Определение множеств для модели
    def read_sets(self):
        self.BCQT = self.db['BQC']['BQC'].values
        self.C = self.db['C']['Component name'].values
        self.D = self.db['D']['Day number'].values
        self.N = self.db['N']['Time interval'].values
        self.P = self.db['P']['Product name'].values
        self.Q = self.db['Q']['Quality name'].values
        self.R = self.db['R']['Reservoir name'].values
        self.T = self.db['T']['Tank name'].values

    # Определение параметров для модели
    def read_parameters(self):
        self.C_cost = self.parameter(self.db["C_cost"], self.C, False)
        self.map_PC = self.parameter(self.db["map_PC"], list(itertools.product(self.P, self.C)), True)
        self.C_init = self.parameter(self.db["C_init"], self.C, False)
        self.C_stock_max = self.parameter(self.db["C_stock_max"], self.C, False)
        self.C_pump_max = self.parameter(self.db["C_pump_max"], self.C, False)
        self.C_pump_min = self.parameter(self.db["C_pump_min"], self.C, False)
        self.CQ = self.parameter(self.db["CQ"], list(itertools.product(self.C, self.Q, self.D)), False)
        self.CQ_init = self.parameter(self.db["CQ_init"], list(itertools.product(self.C, self.Q)), False)
        self.PD_plan = self.parameter(self.db["PD_plan"], list(itertools.product(self.D, self.P)), False)
        self.G = self.parameter(self.db["G"], self.P, True)
        self.P_blend_min = self.parameter(self.db["P_blend_min"], self.P, False)
        self.P_cost = self.parameter(self.db["P_cost"], self.P, False)
        self.P_init = self.parameter(self.db["P_init"], self.P, False)
        self.PQ_max = self.parameter(self.db["PQ_max"],
                                     list(itertools.product(self.P, self.Q, self.D)), False)
        self.PQ_min = self.parameter(self.db["PQ_min"],
                                     list(itertools.product(self.P, self.Q, self.D)), False)
        self.map_CR = self.parameter(self.db["map_CR"], list(itertools.product(self.C, self.R)), True)
        self.R_max = self.parameter(self.db["R_max"], self.R, False)
        self.R_min = self.parameter(self.db["R_min"], self.R, False)
        self.R_pump_max = self.parameter(self.db["R_pump_max"], self.R, False)
        self.R_pump_min = self.parameter(self.db["R_pump_min"], self.R, False)
        self.S = self.parameter(self.db["S"], list(itertools.product(self.D, self.C)), False)
        self.map_PT = self.parameter(self.db["map_PT"], list(itertools.product(self.P, self.T)), True)
        self.T_max = self.parameter(self.db["T_max"], self.T, False)
        self.T_min = self.parameter(self.db["T_min"], self.T, False)
        self.T_pump_max = self.parameter(self.db["T_pump_max"], self.T, False)
        self.T_pump_min = self.parameter(self.db["T_pump_min"], self.T, False)
        self.QBM = self.parameter(self.db["QBM"], list(itertools.product(self.Q, self.BCQT)), True)
        self.Q_i = self.parameter(self.db["Q_i"], self.Q, False)
        self.Q_lb = self.parameter(self.db["Q_lb"],
                                   list(itertools.product(self.C, self.Q, self.P)), False)
        self.Q_vc = self.parameter(self.db["Q_vc"], self.Q, True)
        self.P_passport_time = self.parameter(self.db["P_passport_time"], self.P, False)
        self.P_prepare_time = self.parameter(self.db["P_prepare_time"], self.P, False)

        self.ND = self.parameter(self.db["ND"], list(itertools.product(self.N, self.D)), True)
        self.PN_max = self.parameter(self.db["PN_max"], list(itertools.product(self.N, self.P)), False)
        self.PN_min = self.parameter(self.db["PN_min"], list(itertools.product(self.N, self.P)), False)
        self.PT_stock_init = self.parameter(self.db["PT_stock_init"], list(itertools.product(self.P, self.T)), False)
        # self.t_dead_stock = None
        # self.p_loss = None
        # self.q_calc = None
        # self.c_r_stock = None
        # self.p_q_t_init = None
        # self.c_q_r_init = None

    # def scalars_definition(self):
    # self.ver_w_price_MT = self.scalar(self.db['verify_wight_price_MT'])
    # self.ver_w_price = self.scalar(self.db['verify_wight_price'])
    # self.ver_price_spec = self.scalar(self.db['verify_price_spec'])
    # self.giveaway_price = self.scalar(self.db['giveaway_price'])
    # self.stock_ver_price = self.scalar(self.db['stock_verify_price'])
    # self.safe_stock = self.scalar(self.db['safe_stock'])
    # self.blend_data = self.scalar(self.db['blend_data'])

    @staticmethod
    def parameter(param: pd.DataFrame, domains: list, binary=False):
        """
        :param param: Запись из базы с данными параметра
        :param domains: Домены, на основе которых задан параметр
        :param binary: Определяет тип параметра (бинарный, не бинарный)
        :return: переменная словаря, содержащая данные по параметру, где пустые поля добавлены с нулевым значением
        """

        if len(param.columns) == 2:
            # print(param.values)
            temp = dict((rec[0], rec[1]) for rec in param.values)
        else:
            # print(param.values)
            temp = dict((tuple(rec[:-1]), rec[-1]) for rec in param.values)
        if binary:
            return {rec: 1 if rec in temp else 0 for rec in domains}
        else:
            return {rec: temp[rec] if rec in temp else 0 for rec in domains}

    @staticmethod
    def scalar(param):
        return param.values[0]

    def calc_q(self):
        for q in self.Q:
            for d in self.D:
                for c in self.C:
                    self.CQ_f[c, q, d] = self.CQ[c, q, d] if not self.QBM[q, 'index'] else \
                        self.CQ[c, q, d] ** self.Q_i[q]
                for p in self.P:
                    self.PQ_min_f[p, q, d] = self.PQ_min[p, q, d] if not self.QBM[q, 'index'] else \
                        self.PQ_min[p, q, d] ** self.Q_i[q]
                    self.PQ_max_f[p, q, d] = self.PQ_max[p, q, d] if not self.QBM[q, 'index'] else \
                        self.PQ_max[p, q, d] ** self.Q_i[q]
                    # if q == 'qRCL' and self.PQ_min_f[p, q, d] != 0:
                    #     self.PQ_max_f[p, q, d] = self.PQ_min_f[p, q, d] + self.rcl_gap


class ModelVars:
    def __init__(self):
        """
        ModelVars --- class for variable definition. It stores all variables for convenient use in model.
        """

        # Blend variables
        # blend_comp_activity(d, b, c, p)
        self.C_feed = tupledict()

        # blend_comp_vol(d, b, c, p)
        self.C_feed_vol = tupledict()

        # blend_weight(d, b, p)
        self.P_weight = tupledict()

        # blend_vol(d, b, p)
        self.P_vol = tupledict()

        # blended_weight(d, b, c)
        self.C_weight = tupledict()

        # Stock variables
        # stock_comp(d, b, c)
        self.C_stock = tupledict()

        # Blend time section
        # blend_duration(d, b)
        self.blend_duration = tupledict()

        # Binary variables
        # b_C(d, b, c)
        self.b_C = tupledict()

        # b_P(d, b, p)
        self.b_P = tupledict()

        # Verification section
        # verify_reserve_def(c)
        # self.deviation_reserve_def = tupledict()
        #
        # # verify_reserve_exc(c)
        # self.deviation_reserve_exc = tupledict()

        # verify_supply(d, b, c)
        self.deviation_supply = tupledict()

        # verify_product_def(d, p)
        self.deviation_PD_plan_def = tupledict()

        # verify_product_exc(d, p)
        self.deviation_PD_plan_exc = tupledict()

        # verify_product(n, p)
        self.deviation_PN_plan = tupledict()

        # verify_product_exc(n, p)
        # self.deviation_PN_plan_exc = tupledict()

        # # verify_component_def(c)
        # self.v_component_def = tupledict()
        #
        # # verify_component_exc(c)
        # self.v_component_exc = tupledict()

        # verify_spec(b, q, p)
        self.deviation_PQ_spec = tupledict()

        # verify_spec_RCL(b, quality, blend_product)
        self.deviation_PQ_spec_rcl = tupledict()

        # Penalty Section
        # penalty_supply
        self.penalty_supply = None

        # penalty_reserve
        # self.penalty_reserve = None

        # penalty_product
        self.penalty_PD_plan = None
        self.penalty_PN_plan = None

        # # penalty_component
        # self.penalty_component = None

        # penalty_spec
        self.penalty_specification = None

        # Economic section
        self.C_total_cost = None
        self.P_total_cost = None
        self.profit = None


def add_variables(m: Model, data: ModelData, v: ModelVars):
    # Blend variables
    v.C_feed = tupledict({(d, b, c, p): m.addVar(lb=0.0,
                                                 ub=COPT.INFINITY if b % 2 == 0 else 0.0,
                                                 vtype=COPT.CONTINUOUS,
                                                 name="C_feed({0},{1},{2},{3})".format(
                                                     d, b, c, p)) for d in data.D for b in data.B
                          for c in data.C for p in data.P})

    v.C_feed_vol = tupledict({(d, b, c, p): m.addVar(lb=0.0,
                                                     ub=COPT.INFINITY if b % 2 == 0 else 0.0,
                                                     vtype=COPT.CONTINUOUS,
                                                     name="C_feed_vol({0},{1},{2},{3})".format(
                                                         d, b, c, p)) for d in data.D for b in data.B
                              for c in data.C for p in data.P})

    v.P_weight = tupledict({(d, b, p): m.addVar(lb=0.0,
                                                ub=COPT.INFINITY if b % 2 == 0 else 0.0,
                                                vtype=COPT.CONTINUOUS,
                                                name="P_weight({0},{1},{2})".format(d, b, p)) for
                            p in data.P for b in data.B for d in data.D})

    v.P_vol = tupledict({(d, b, p): m.addVar(lb=0.0,
                                             ub=COPT.INFINITY if b % 2 == 0 else 0.0,
                                             vtype=COPT.CONTINUOUS,
                                             name="P_vol({0},{1},{2})".format(d, b, p)) for
                         p in data.P for b in data.B for d in data.D})

    v.C_weight = tupledict({(d, b, c): m.addVar(lb=0.0,
                                                ub=COPT.INFINITY if b % 2 == 0 else 0.0,
                                                vtype=COPT.CONTINUOUS,
                                                name="C_weight({0},{1},{2})".format(d, b, c))
                            for d in data.D for b in data.B for c in data.C})

    # Stock variables
    v.C_stock = tupledict({(d, b, c): m.addVar(lb=0.0,
                                               ub=data.C_stock_max[c],
                                               vtype=COPT.CONTINUOUS,
                                               name="C_stock({0},{1},{2})".format(d, b, c))
                           for d in data.D for b in data.B for c in data.C})

    # Blend time section
    v.blend_duration = tupledict({(d, b): m.addVar(lb=0.0,
                                                   ub=24,
                                                   vtype=COPT.CONTINUOUS,
                                                   name="blend_duration({0},{1})".format(d, b))
                                  for b in data.B for d in data.D})

    # Binary variables
    v.b_C = tupledict({(d, b, c): m.addVar(lb=0.0,
                                           ub=1 if b % 2 == 0 else 0.0,
                                           vtype=COPT.BINARY,
                                           name="b_C({0},{1},{2})".format(d, b, c))
                       for d in data.D for b in data.B for c in data.C})

    v.b_P = tupledict({(d, b, p): m.addVar(lb=0.0,
                                           ub=1 if b % 2 == 0 else 0.0,
                                           vtype=COPT.BINARY,
                                           name="b_P({0},{1},{2})".format(d, b, p))
                       for d in data.D for b in data.B for p in data.P})

    # Verification section
    # v.v_reserve_def = tupledict({c: m.addVar(lb=0.0,
    #                                          ub=COPT.INFINITY,
    #                                          vtype=COPT.CONTINUOUS,
    #                                          name="verify_reserve_def({0})".format(c))
    #                              for c in data.C})
    #
    # v.v_reserve_exc = tupledict({c: m.addVar(lb=0.0,
    #                                          ub=COPT.INFINITY,
    #                                          vtype=COPT.CONTINUOUS,
    #                                          name="verify_reserve_exc({0})".format(c))
    #                              for c in data.C})

    v.deviation_supply = tupledict({(d, b, c): m.addVar(lb=0.0,
                                                        ub=COPT.INFINITY,
                                                        vtype=COPT.CONTINUOUS,
                                                        name="deviation_supply({0},{1},{2})".format(d, b, c))
                                    for d in data.D for b in data.B for c in data.C})

    v.deviation_PD_plan_def = tupledict({(d, p): m.addVar(lb=0.0,
                                                          ub=0.0,
                                                          # ub=COPT.INFINITY,
                                                          vtype=COPT.CONTINUOUS,
                                                          name="deviation_PD_plan_def({0},{1})".format(d, p))
                                         for d in data.D for p in data.P})

    v.deviation_PD_plan_exc = tupledict({(d, p): m.addVar(lb=0.0,
                                                          ub=COPT.INFINITY,
                                                          vtype=COPT.CONTINUOUS,
                                                          name="deviation_PD_plan_exc({0},{1})".format(d, p))
                                         for d in data.D for p in data.P})

    v.deviation_PN_plan = tupledict({(n, p): m.addVar(lb=0.0,
                                                      ub=0.0,
                                                      # ub=COPT.INFINITY,
                                                      vtype=COPT.CONTINUOUS,
                                                      name="deviation_PN_plan({0},{1})".format(n, p))
                                     for n in data.N for p in data.P})

    # v.deviation_PN_plan_exc = tupledict({(n, p): m.addVar(lb=0.0,
    #                                                       ub=COPT.INFINITY,
    #                                                       vtype=COPT.CONTINUOUS,
    #                                                       name="deviation_PN_plan_exc({0},{1})".format(n, p))
    #                                      for n in data.N for p in data.P})

    # v.v_component_def = tupledict({c: m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
    #                                             name="verify_component_def({0})".format(c))
    #                                for c in data.C})
    #
    # v.v_component_exc = tupledict({c: m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
    #                                             name="verify_component_exc({0})".format(c))
    #                                for c in data.C})

    v.deviation_PQ_spec = tupledict({(d, b, p, q): m.addVar(lb=0.0,
                                                            ub=COPT.INFINITY,
                                                            vtype=COPT.CONTINUOUS,
                                                            name="deviation_PQ_spec({0},{1},{2},{3})".format(d, b, p,
                                                                                                             q))
                                     for d in data.D for b in data.B for p in data.P for q in data.Q})

    # v.v_spec_rcl = tupledict({('qRCL', bp): m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
    #                                                  name="verify_spec_rcl({0},{1})".format('qRCL', bp))
    #                           for bp in data.bp})

    # Penalty Section
    v.penalty_supply = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                                name="penalty_supply")

    # v.penalty_reserve = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
    #                              name="penalty_reserve")

    v.penalty_PD_plan = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                                 name="penalty_PD_plan")

    v.penalty_PN_plan = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                                 name="penalty_PN_plan")

    # v.penalty_component = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
    #                                name="penalty_component")

    v.penalty_specification = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                                       name="penalty_specification")

    v.C_total_cost = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                              name="C_total_cost")
    v.P_total_cost = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                              name="P_total_cost")
    v.profit = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                        name="profit")


def gantt_diagram(model: Model, data: ModelData, v: ModelVars, result: dict):
    """
    Drivers Gantt diagram plotting function. Shows the optimal driver schedule on the timeline.
    :param model:
    :param data:
    :param v:
    :param result:
    :return: None
    """

    gp_tags = [p for p in data.P if data.G[p]]
    time_horizon = data.H
    fig1, gnt1 = plt.subplots(figsize=(15, 3))
    # gnt1.grid(True)
    curr_time = 0
    for d in data.D:
        for b in data.B:
            for pi, p in enumerate(gp_tags):
                # if data.G[p] == 1:
                if result['b_P'][d, b, p] > 0.5:
                    # print(f'b_P({d},{b},{p}) = 1')
                    # print(result['blend_duration'][d, b])
                    gnt1.broken_barh([(curr_time, result['blend_duration'][d, b])],
                                     (pi + 1 - 1 / 8, 1 / 4), edgecolors='black', facecolors='white', hatch='////')
            curr_time += result['blend_duration'][d, b]

    plt.title("График приготовления продуктов")
    gnt1.set_xlim(-1, time_horizon + 1)
    gnt1.set_ylim(0, 1 * len(gp_tags) + 1 / 2)
    gnt1.set_xlabel('Время, час')
    gnt1.set_ylabel('Продукт ' + r'$P$')
    gnt1.set_xticks(range(0, time_horizon + 1, 24))
    gnt1.set_yticks([i * 1 for i in range(1, len(gp_tags) + 1)])
    gnt1.set_xticklabels(range(0, time_horizon + 1, 24))
    gnt1.set_yticklabels(gp_tags)
    plt.tight_layout()
    plt.savefig(data.result_path + "/pictures/product_gannt.pdf", format="pdf")
    plt.show()

    fig2, gnt2 = plt.subplots(figsize=(15, 3))
    # gnt2.grid(True)
    curr_time = 0
    for d in data.D:
        for b in data.B:
            for ci, c in enumerate(data.C):
                # if data.G[p] == 1:
                if result['b_C'][d, b, c] > 0.5:
                    # print(f'b_C({d},{b},{c}) = 1')
                    # print(result['blend_duration'][d, b])
                    gnt2.broken_barh([(curr_time, result['blend_duration'][d, b])],
                                     (ci + 1 - 1 / 4, 1 / 2), edgecolors='black', facecolors='white', hatch='//')
            curr_time += result['blend_duration'][d, b]

    plt.title("График использования компонентов".format(d))
    gnt2.set_xlim(-1, time_horizon + 1)
    gnt2.set_ylim(0, 1 * len(data.C) + 1 / 2)
    gnt2.set_xlabel('Время, час')
    gnt2.set_ylabel('Компонент ' + r'$C$')
    gnt2.set_xticks(range(0, time_horizon + 1, 24))
    gnt2.set_yticks([i * 1 for i in range(1, len(data.C) + 1)])
    gnt2.set_xticklabels(range(0, time_horizon + 1, 24))
    gnt2.set_yticklabels(data.C)
    plt.tight_layout()
    plt.savefig(data.result_path + "/pictures/component_gannt.pdf", format="pdf")
    plt.show()


def get_var_values(m: Model):
    """
    Get variables from the optimal solution
    :param m: model to extract solution data
    :param varlist:
    :return: optimal solution data
    """
    all_vars = m.getVars()
    var_names = []
    var_set = {}
    for v in all_vars:
        v_full_name = v.getName()
        if '(' in v_full_name:
            v_name, v_domains = v_full_name.split('(')
            v_domains = v_domains[:-1].split(',')
            var_names.append(v_name)
            if var_set.get(v_name) is None:
                var_set[v_name] = len(v_domains)
        else:
            var_names.append(v_full_name)
            if var_set.get(v_full_name) is None:
                var_set[v_full_name] = 0

    # var_dict = {var: pd.DataFrame(columns=['Column' + str(i + 1)
    #                                        for i in range(var_set[var])] + ['Value']) for var in var_set.keys()}
    var_dict = {var: pd.DataFrame() for var in var_set.keys()}
    for v in all_vars:
        v_full_name = v.getName()
        if '(' in v_full_name:
            v_name, v_domains = v_full_name.split('(')
            v_domains = [int(s) if s.isdigit() else s for s in v_domains[:-1].split(',')]
            if var_dict.get(v_name) is not None:
                # var_dict[v_name].append(v_domains + [v.x])
                var_dict[v_name] = pd.concat([var_dict[v_name],
                                              pd.Series(v_domains + [v.x, v.getInfo(COPT.Info.LB),
                                                                     v.getInfo(COPT.Info.UB)]).to_frame().T],
                                             axis=0,
                                             ignore_index=True)
        else:
            if var_dict.get(v_full_name) is not None:
                # var_dict[v_full_name].append([v.x])
                var_dict[v_full_name] = pd.concat([var_dict[v_full_name],
                                                   pd.Series([v.x, v.getInfo(COPT.Info.LB),
                                                              v.getInfo(COPT.Info.UB)]).to_frame().T],
                                                  axis=0,
                                                  ignore_index=True)

    var_dict_records = {}
    for var in var_set.keys():
        if var_set[var] > 0:
            # print(list(var_dict[var].columns)[:var_set[var]])
            var_dict[var].sort_values(by=list(var_dict[var].columns)[:var_set[var]], inplace=True)

        var_dict[var].columns = ['Column' + str(i + 1) for i in range(var_set[var])] + ['Value', 'Lower bound',
                                                                                        'Upper bound']
        records = var_dict[var].loc[:, :'Value']
        # print(records.head())
        # print(var)
        if var_set[var] > 0:
            var_dict_records[var] = records.set_index(['Column' + str(i + 1) for
                                                       i in range(var_set[var])]).to_dict()['Value']
        else:
            var_dict_records[var] = records['Value'].values

    var_dict['objective'] = pd.DataFrame(data=[m.objval], columns=['Value'])

    return var_dict, var_dict_records


def sol2excel(result, run_path, db: dict = None):
    # Creating Excel Writer Object from Pandas
    with pd.ExcelWriter(run_path + '/result.xlsx', engine='xlsxwriter') as writer:
        workbook = writer.book
        if db is not None:
            for param in db.keys():
                worksheet = workbook.add_worksheet(param)
                writer.sheets[param] = worksheet
                db[param].to_excel(writer, sheet_name=param, startrow=0, startcol=0, index=False)

                # Get the dimensions of the dataframe.
                (max_row, max_col) = db[param].shape

                # Make the columns wider for clarity.
                worksheet.set_column(0, max_col - 1, 20)

                # Set the autofilter.
                worksheet.autofilter(0, 0, max_row, max_col - 1)

        for variable in result.keys():
            worksheet = workbook.add_worksheet(variable)
            writer.sheets[variable] = worksheet
            result[variable].to_excel(writer, sheet_name=variable, startrow=0, startcol=0, index=False)

            # Get the dimensions of the dataframe.
            (max_row, max_col) = result[variable].shape

            # Make the columns wider for clarity.
            worksheet.set_column(0, max_col - 1, 20)

            # Set the autofilter.
            worksheet.autofilter(0, 0, max_row, max_col - 1)


'''
def auto_gannt(result: dict):
    """
    Drivers Gantt diagram plotting function. Shows the optimal driver schedule on the timeline.
    :param model:
    :param data:
    :param v:
    :param result:
    :return: None
    """

    gp_tags = [p for p in data.P if data.G[p]]
    time_horizon = data.H
    for d in data.D:
        fig, gnt = plt.subplots(figsize=(15, 6))
        gnt.grid(True)
        curr_time = 0
        for b in data.B:
            for pi, p in enumerate(gp_tags):
                # if data.G[p] == 1:
                if result['b_prod'][d, b, p] > 0.5:
                    print(f'b_prod({d},{b},{p}) = 1')
                    print(result['blend_duration'][d, b])
                    gnt.broken_barh([(curr_time, result['blend_duration'][d, b])],
                                    (pi + 1 - 1 / 4, 1 / 2), facecolors='blue')
            curr_time += result['blend_duration'][d, b]

        plt.title("График приготовления продуктов на день {0}".format(d))
        gnt.set_xlim(-1, 24 + 1)
        gnt.set_ylim(0, 1 * len(gp_tags) + 1 / 2)
        gnt.set_xlabel('Время, час')
        gnt.set_ylabel('Продукт ' + r'$(P)$')
        gnt.set_xticks(range(0, 24 + 1, 2))
        gnt.set_yticks([i * 1 for i in range(1, len(gp_tags) + 1)])
        gnt.set_xticklabels(range(0, 24 + 1, 2))
        gnt.set_yticklabels(gp_tags)
        plt.tight_layout()
        plt.savefig(data.result_path + "/pictures/product_gannt_{0}_day.pdf".format(d),
                    format="pdf")
        plt.show()

        fig, gnt = plt.subplots(figsize=(15, 6))
        gnt.grid(True)
        curr_time = 0
        for b in data.B:
            for ci, c in enumerate(data.C):
                # if data.G[p] == 1:
                if result['b_comp'][d, b, c] > 0.5:
                    print(f'b_comp({d},{b},{c}) = 1')
                    print(result['blend_duration'][d, b])
                    gnt.broken_barh([(curr_time, result['blend_duration'][d, b])],
                                    (ci + 1 - 1 / 4, 1 / 2), facecolors='red')
            curr_time += result['blend_duration'][d, b]

        plt.title("График использования компонентов на день {0}".format(d))
        gnt.set_xlim(-1, 24 + 1)
        gnt.set_ylim(0, 1 * len(data.C) + 1 / 2)
        gnt.set_xlabel('Время, час')
        gnt.set_ylabel('Компонент ' + r'$(C)$')
        gnt.set_xticks(range(0, 24 + 1, 2))
        gnt.set_yticks([i * 1 for i in range(1, len(data.C) + 1)])
        gnt.set_xticklabels(range(0, 24 + 1, 2))
        gnt.set_yticklabels(data.C)
        plt.tight_layout()
        plt.savefig(data.result_path + "/pictures/component_gannt_{0}_day.pdf".format(d),
                    format="pdf")
        plt.show()

'''

