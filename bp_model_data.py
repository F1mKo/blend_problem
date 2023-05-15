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
        self.C = None  # component
        self.D = None  # day
        self.H = None  # time horizon
        self.N = None  # time interval
        self.P = None  # product
        self.Q = None  # quality

        # Parameters
        self.CQ = None  # component quality
        self.C_cost = None  # component cost
        self.C_init = None  # component init stock
        self.C_pump_max = None  # maximum component involvement velocity
        self.C_pump_min = None  # minimum component involvement velocity
        self.C_stock_max = None  # component max stock
        self.map_PC = None  # component to product map
        self.ND = None  # day to time interval
        self.PD_plan = None  # product day plan
        self.PN_max = None  # maximum product time interval plan
        self.PN_min = None  # minimum product time interval plan
        self.PQ_max = None  # maximum product quality specification
        self.PQ_min = None  # minimum product quality specification
        self.P_blend_min = None  # minimum product blend weight
        self.P_cost = None  # product cost
        self.P_prepare_time = None  # product quality checking time

        self.Q_vc = None  # volumetric calculation of quality
        self.S = None  # component supply

        self.CQ_f = {}
        self.PQ_max_f = {}
        self.PQ_min_f = {}
        self.B = None
        self.b2d = None

        self.big_m = None

    def import_data_excel(self, config, result_path):
        # get data from Excel file
        self.H = config['time_horizon']
        self.model_name = config['Model name']
        self.result_path = result_path
        self.db = pd.read_excel('init_data/' + config['input_file'], sheet_name=None)
        self.big_m = 1E5

        self.read_sets()
        self.read_parameters()
        self.calc_q()
        self.create_params()

    def create_params(self):
        self.S = {domains: self.S[domains] / 24 for domains in self.S.keys()}
        self.B = [b + 1 for b in range(2 * len(self.P) + 1)]
        self.b2d = {(b, d): 1 for b in self.B for d in self.D}

    # Определение множеств для модели
    def read_sets(self):
        self.C = self.db['C']['Component name'].values
        self.D = self.db['D']['Day number'].values
        self.N = self.db['N']['Time interval'].values
        self.P = self.db['P']['Product name'].values
        self.Q = self.db['Q']['Quality name'].values

    # Определение параметров для модели
    def read_parameters(self):
        self.CQ = self.parameter(self.db["CQ"], list(itertools.product(self.C, self.Q, self.D)), False)
        self.C_cost = self.parameter(self.db["C_cost"], self.C, False)
        self.C_init = self.parameter(self.db["C_init"], self.C, False)
        self.C_pump_max = self.parameter(self.db["C_pump_max"], self.C, False)
        self.C_pump_min = self.parameter(self.db["C_pump_min"], self.C, False)
        self.C_stock_max = self.parameter(self.db["C_stock_max"], self.C, False)
        self.map_PC = self.parameter(self.db["map_PC"], list(itertools.product(self.P, self.C)), True)
        self.ND = self.parameter(self.db["ND"], list(itertools.product(self.N, self.D)), True)
        self.PD_plan = self.parameter(self.db["PD_plan"], list(itertools.product(self.D, self.P)), False)
        self.PN_max = self.parameter(self.db["PN_max"], list(itertools.product(self.N, self.P)), False)
        self.PN_min = self.parameter(self.db["PN_min"], list(itertools.product(self.N, self.P)), False)
        self.PQ_max = self.parameter(self.db["PQ_max"],
                                     list(itertools.product(self.P, self.Q, self.D)), False)
        self.PQ_min = self.parameter(self.db["PQ_min"],
                                     list(itertools.product(self.P, self.Q, self.D)), False)
        self.P_blend_min = self.parameter(self.db["P_blend_min"], self.P, False)
        self.P_cost = self.parameter(self.db["P_cost"], self.P, False)
        self.P_prepare_time = self.parameter(self.db["P_prepare_time"], self.P, False)
        self.Q_vc = self.parameter(self.db["Q_vc"], self.Q, True)
        self.S = self.parameter(self.db["S"], list(itertools.product(self.D, self.C)), False)

    @staticmethod
    def parameter(param: pd.DataFrame, domains: list, binary=False):
        """
        :param param: Запись из базы с данными параметра
        :param domains: Домены, на основе которых задан параметр
        :param binary: Определяет тип параметра (бинарный, не бинарный)
        :return: переменная словаря, содержащая данные по параметру, где пустые поля добавлены с нулевым значением
        """

        if len(param.columns) == 2:
            temp = dict((rec[0], rec[1]) for rec in param.values)
        else:
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
                    self.CQ_f[c, q, d] = self.CQ[c, q, d] if q != 'rvp' else \
                        self.CQ[c, q, d] ** 1.25
                for p in self.P:
                    self.PQ_min_f[p, q, d] = self.PQ_min[p, q, d] if q != 'rvp' else \
                        self.PQ_min[p, q, d] ** 1.25
                    self.PQ_max_f[p, q, d] = self.PQ_max[p, q, d] if q != 'rvp' else \
                        self.PQ_max[p, q, d] ** 1.25


class ModelVars:
    def __init__(self):
        """
        ModelVars --- class for variable definition. It stores all variables for convenient use in model.
        """

        # Blend variables
        self.C_feed = tupledict()
        self.C_feed_vol = tupledict()
        self.P_weight = tupledict()
        self.P_volume = tupledict()
        self.C_weight = tupledict()

        # Stock variables
        self.C_stock = tupledict()

        # Blend time section
        self.b_duration = tupledict()

        # Binary variables
        self.b_C = tupledict()
        self.b_P = tupledict()

        # Verification section
        self.dev_stock = tupledict()
        self.dev_plan_PD_def = tupledict()
        self.dev_plan_PD_exc = tupledict()
        self.dev_plan_PN = tupledict()
        self.dev_spec_PQ = tupledict()

        # Penalty Section
        self.pen_stock = None
        self.pen_plan_PD = None
        self.pen_plan_PN = None
        self.pen_spec_PQ = None

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

    v.P_volume = tupledict({(d, b, p): m.addVar(lb=0.0,
                                                ub=COPT.INFINITY if b % 2 == 0 else 0.0,
                                                vtype=COPT.CONTINUOUS,
                                                name="P_volume({0},{1},{2})".format(d, b, p)) for
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
    v.b_duration = tupledict({(d, b): m.addVar(lb=0.0,
                                               ub=24,
                                               vtype=COPT.CONTINUOUS,
                                               name="b_duration({0},{1})".format(d, b))
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

    # Slack section
    v.dev_stock = tupledict({(d, b, c): m.addVar(lb=0.0,
                                                 ub=COPT.INFINITY,
                                                 vtype=COPT.CONTINUOUS,
                                                 name="dev_stock({0},{1},{2})".format(d, b, c))
                             for d in data.D for b in data.B for c in data.C})

    v.dev_plan_PD_def = tupledict({(d, p): m.addVar(lb=0.0,
                                                    ub=COPT.INFINITY,
                                                    vtype=COPT.CONTINUOUS,
                                                    name="dev_plan_PD_def({0},{1})".format(d, p))
                                   for d in data.D for p in data.P})

    v.dev_plan_PD_exc = tupledict({(d, p): m.addVar(lb=0.0,
                                                    ub=COPT.INFINITY,
                                                    vtype=COPT.CONTINUOUS,
                                                    name="dev_plan_PD_exc({0},{1})".format(d, p))
                                   for d in data.D for p in data.P})

    v.dev_plan_PN = tupledict({(n, p): m.addVar(lb=0.0,
                                                ub=COPT.INFINITY,
                                                vtype=COPT.CONTINUOUS,
                                                name="dev_plan_PN({0},{1})".format(n, p))
                               for n in data.N for p in data.P})

    v.dev_spec_PQ = tupledict({(d, b, p, q): m.addVar(lb=0.0,
                                                      ub=COPT.INFINITY,
                                                      vtype=COPT.CONTINUOUS,
                                                      name="dev_spec_PQ({0},{1},{2},{3})".format(d, b, p,
                                                                                                 q))
                               for d in data.D for b in data.B for p in data.P for q in data.Q})

    # Penalty Section
    v.pen_stock = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                           name="pen_stock")

    v.pen_plan_PD = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                             name="pen_plan_PD")

    v.pen_plan_PN = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                             name="pen_plan_PN")

    v.pen_spec_PQ = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                             name="pen_spec_PQ")

    v.C_total_cost = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                              name="C_total_cost")
    v.P_total_cost = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                              name="P_total_cost")
    v.profit = m.addVar(lb=0.0, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS,
                        name="profit")


def gantt_diagram(data: ModelData, result: dict):
    """
    Drivers Gantt diagram plotting function. Shows the optimal driver schedule on the timeline.
    :param data:
    :param result:
    :return: None
    """

    time_horizon = data.H
    fig1, gnt1 = plt.subplots(figsize=(15, 3))
    # gnt1.grid(True)
    curr_time = 0
    for d in data.D:
        for b in data.B:
            for pi, p in enumerate(data.P):
                if result['b_P'][d, b, p] > 0.5:
                    gnt1.broken_barh([(curr_time, result['b_duration'][d, b])],
                                     (pi + 1 - 1 / 8, 1 / 4), edgecolors='black', facecolors='white', hatch='////')
            curr_time += result['b_duration'][d, b]

    plt.title("График приготовления продуктов")
    gnt1.set_xlim(-1, time_horizon + 1)
    gnt1.set_ylim(0, 1 * len(data.P) + 1 / 2)
    gnt1.set_xlabel('Время, час')
    gnt1.set_ylabel('Продукт ' + r'$P$')
    gnt1.set_xticks(range(0, time_horizon + 1, 24))
    gnt1.set_yticks([i * 1 for i in range(1, len(data.P) + 1)])
    gnt1.set_xticklabels(range(0, time_horizon + 1, 24))
    gnt1.set_yticklabels(data.P)
    plt.tight_layout()
    plt.savefig(data.result_path + "/pictures/product_gannt.pdf", format="pdf")
    plt.show()

    fig2, gnt2 = plt.subplots(figsize=(15, 3))
    # gnt2.grid(True)
    curr_time = 0
    for d in data.D:
        for b in data.B:
            for ci, c in enumerate(data.C):

                if result['b_C'][d, b, c] > 0.5:
                    gnt2.broken_barh([(curr_time, result['b_duration'][d, b])],
                                     (ci + 1 - 1 / 4, 1 / 2), edgecolors='black', facecolors='white', hatch='//')
            curr_time += result['b_duration'][d, b]

    plt.title("График использования компонентов")
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

    var_dict = {var: pd.DataFrame() for var in var_set.keys()}
    for v in all_vars:
        v_full_name = v.getName()
        if '(' in v_full_name:
            v_name, v_domains = v_full_name.split('(')
            v_domains = [int(s) if s.isdigit() else s for s in v_domains[:-1].split(',')]
            if var_dict.get(v_name) is not None:
                var_dict[v_name] = pd.concat([var_dict[v_name],
                                              pd.Series(v_domains + [v.x, v.getInfo(COPT.Info.LB),
                                                                     v.getInfo(COPT.Info.UB)]).to_frame().T],
                                             axis=0,
                                             ignore_index=True)
        else:
            if var_dict.get(v_full_name) is not None:
                var_dict[v_full_name] = pd.concat([var_dict[v_full_name],
                                                   pd.Series([v.x, v.getInfo(COPT.Info.LB),
                                                              v.getInfo(COPT.Info.UB)]).to_frame().T],
                                                  axis=0,
                                                  ignore_index=True)

    var_dict_records = {}
    for var in var_set.keys():
        if var_set[var] > 0:
            var_dict[var].sort_values(by=list(var_dict[var].columns)[:var_set[var]], inplace=True)

        var_dict[var].columns = ['Column' + str(i + 1) for i in range(var_set[var])] + ['Value', 'Lower bound',
                                                                                        'Upper bound']
        records = var_dict[var].loc[:, :'Value']
        if var_set[var] > 0:
            var_dict_records[var] = records.set_index(['Column' + str(i + 1) for
                                                       i in range(var_set[var])]).to_dict()['Value']
        else:
            var_dict_records[var] = records['Value'].values

    var_dict['objective'] = pd.DataFrame(data=[m.objval], columns=['Value'])

    return var_dict, var_dict_records


def sol2excel(result, run_path, db: dict = None, folder_name: str = None):
    # Creating Excel Writer Object from Pandas
    with pd.ExcelWriter(run_path + '/' + (folder_name if folder_name is not None else '') +
                        '_result.xlsx', engine='xlsxwriter') as writer:
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

                # Set the auto-filter.
                worksheet.autofilter(0, 0, max_row, max_col - 1)

        for variable in result.keys():
            worksheet = workbook.add_worksheet(variable)
            writer.sheets[variable] = worksheet
            result[variable].to_excel(writer, sheet_name=variable, startrow=0, startcol=0, index=False)

            # Get the dimensions of the dataframe.
            (max_row, max_col) = result[variable].shape

            # Make the columns wider for clarity.
            worksheet.set_column(0, max_col - 1, 20)

            # Set the auto-filter.
            worksheet.autofilter(0, 0, max_row, max_col - 1)
