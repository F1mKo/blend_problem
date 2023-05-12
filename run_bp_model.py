from coptpy import *
from bp_model_data import output_folder_check, ModelData, ModelVars, add_variables, gantt_diagram, \
    get_var_values, sol2excel


config = {
    'project_path': 'C:/Users/F1mK0/PycharmProjects/copt_env',
    'input_file': 'bp_rendat_NoRVI_new_supply_new_quality.xlsx',
    'Model name': 'Blend Problem',
    'time_horizon': 288
}


def gasoline_logic_def(m: Model, data: ModelData, v: ModelVars):
    for p in data.P:
        # blend weight min constraint
        pw_min_c = tupledict({(d, b, p): m.addConstr(lhs=v.P_weight[d, b, p],
                                                     sense=COPT.GREATER_EQUAL,
                                                     rhs=data.P_blend_min[p] * v.b_P[d, b, p],
                                                     name='P_weight_min_constr({0},{1},{2})'.format(d, b, p))
                              for d in data.D for b in data.B})

        # blend weight max constraint
        pw_max_c = tupledict({(d, b, p): m.addConstr(lhs=v.P_weight[d, b, p],
                                                     sense=COPT.LESS_EQUAL,
                                                     rhs=data.big_m * v.b_P[d, b, p],
                                                     name='P_weight_max_constr({0},{1},{2})'.format(d, b, p))
                              for d in data.D for b in data.B})

        # dev_PD_def_fix = tupledict({(d, p): m.addConstr(lhs=v.deviation_PD_plan_def[d, p],
        #                                                 sense=COPT.EQUAL,
        #                                                 rhs=0,
        #                                                 name='deviation_PD_plan_def({0},{1})'.format(d, p))
        #                             for d in data.D})

    # one product for batch
    op2b_c = tupledict(
        {(d, b): m.addConstr(lhs=quicksum(v.b_P[d, b, p] for p in data.P),
                             sense=COPT.LESS_EQUAL,
                             rhs=1, name='one_p_for_b({0},{1})'.format(d, b))
         for d in data.D for b in data.B})

    # one batch for product
    ob2p_c = tupledict({(d, p): m.addConstr(lhs=quicksum(v.b_P[d, b, p] for b in data.B),
                                            sense=COPT.LESS_EQUAL,
                                            rhs=1,
                                            name='one_b_for_p({0},{1})'.format(d, p))
                        for d in data.D for p in data.P})

    # b_C_constr
    b_C_constr = tupledict({(d, b): m.addConstr(lhs=quicksum(v.b_C[d, b, c] for c in data.C),
                                                sense=COPT.LESS_EQUAL,
                                                rhs=len(data.C) * quicksum(v.b_P[d, b, p] for p in data.P),
                                                name='b_C_constr({0},{1})'.format(d, b))
                            for d in data.D for b in data.B})

    # blend_duration constraint
    bd_c = tupledict({d: m.addConstr(lhs=v.blend_duration.sum(d, '*'),
                                     sense=COPT.EQUAL,
                                     rhs=24,
                                     name='b_duration_constr({0})'.format(d))
                      for d in data.D})

    # blend_duration constraint 1
    bd_c1 = tupledict({d: m.addConstr(lhs=v.blend_duration[d, b],
                                      sense=COPT.GREATER_EQUAL,
                                      rhs=quicksum(v.b_P[d, b, p] for p in data.P),
                                      name='b_duration_constr_min({0},{1})'.format(d, b))
                       for d in data.D for b in data.B if b % 2 == 0})

    # blend_duration constraint min
    bd_c2 = tupledict({d: m.addConstr(lhs=v.blend_duration[d, b],
                                      sense=COPT.LESS_EQUAL,
                                      rhs=24 * quicksum(v.b_P[d, b, p] for p in data.P),
                                      name='b_duration_constr_max({0}.{1})'.format(d, b))
                       for d in data.D for b in data.B if b % 2 == 0})

    # blend_duration constraint min
    bd_c3 = tupledict({d: m.addConstr(lhs=v.blend_duration[d, b - 1] +
                                          (v.blend_duration[d - 1, data.B[-1]] if d > 1 and b == 2 else 0),
                                      sense=COPT.GREATER_EQUAL,
                                      rhs=quicksum(
                                          v.b_P[d, b, p] * data.P_prepare_time[p] for p in data.P),
                                      name='blend_duration_constr3({0}.{1})'.format(d, b))
                       for d in data.D for b in data.B[1:] if b % 2 == 0})

    # period_prod_constr
    tupledict(
        {(d, b): m.addConstr(lhs=quicksum(v.b_P[d, b, p] for p in data.P if data.G[p]),
                             sense=COPT.LESS_EQUAL,
                             rhs=quicksum(v.b_P[d, b + 2, p] for p in data.P if data.G[p]),
                             name='b_P_constr({0})'.format(d, b))
         for d in data.D for b in data.B[:-2] if b % 2 == 0})


def blend_logic_def(m: Model, data: ModelData, v: ModelVars):
    # product weight definition
    pwd_c = tupledict({(d, b, p): m.addConstr(lhs=v.P_weight[d, b, p],
                                              sense=COPT.EQUAL,
                                              rhs=quicksum(
                                                  v.C_feed[d, b, c, p] for c in data.C if (data.map_PC[p, c] == 1)),
                                              name='P_weight_def({0},{1},{2})'.format(d, b, p))
                       for p in data.P for b in data.B for d in data.D})

    # product volume definition
    pvd_c = tupledict({(d, b, p): m.addConstr(lhs=v.P_vol[d, b, p],
                                              sense=COPT.EQUAL,
                                              rhs=quicksum(
                                                  v.C_feed_vol[d, b, c, p] for c in data.C if (data.map_PC[p, c] == 1)),
                                              name='P_vol_def({0},{1},{2})'.format(d, b, p))
                       for p in data.P for b in data.B for d in data.D})

    # component weight definition
    cwd_c = tupledict(
        {(d, b, c): m.addConstr(lhs=v.C_weight[d, b, c],
                                sense=COPT.EQUAL,
                                rhs=quicksum(v.C_feed[d, b, c, p] for p in data.P if (data.map_PC[p, c] == 1)),
                                name='C_weight_def({0},{1},{2})'.format(d, b, c))
         for d in data.D for b in data.B for c in data.C})

    # component weight to volume constraints
    cw2v_c = tupledict(
        {(d, b, c, p): m.addConstr(lhs=v.C_feed[d, b, c, p],
                                   sense=COPT.EQUAL,
                                   rhs=v.C_feed_vol[d, b, c, p] * data.CQ[c, 'qSPG', d],
                                   name='C_vol_constr({0},{1},{2},{3})'.format(d, b, c, p))
         for d in data.D for b in data.B for c in data.C for p in data.P})

    # blend_comp_activity fix
    ca_fix = tupledict(
        {(d, b, c, p): m.addConstr(lhs=v.C_feed[d, b, c, p],
                                   sense=COPT.EQUAL,
                                   rhs=0,
                                   name='C_a_fix({0},{1},{2},{3})'.format(d, b, c, p))
         for d in data.D for b in data.B for c in data.C for p in data.P if
         (not data.map_PC[p, c] or b % 2 == 1)})


def pump_constr_def(m: Model, data: ModelData, v: ModelVars):
    # pump_min_constr
    tupledict({(d, b, c): m.addConstr(lhs=v.C_weight[d, b, c],
                                      sense=COPT.GREATER_EQUAL,
                                      rhs=data.C_pump_min[c] * data.CQ[c, 'qSPG', d] * (v.blend_duration[d, b] - 24 * (
                                              1 - v.b_C[d, b, c])),
                                      name='C_pump_min({0},{1},{2})'.format(d, b, c))
               for d in data.D for b in data.B for c in data.C})

    # pump_max_constr1
    tupledict({(d, b, c): m.addConstr(lhs=v.C_weight[d, b, c],
                                      sense=COPT.LESS_EQUAL,
                                      rhs=data.C_pump_max[c] * data.CQ[c, 'qSPG', d] * v.blend_duration[d, b],
                                      name='C_pump_max1({0},{1},{2})'.format(d, b, c))
               for d in data.D for b in data.B for c in data.C})

    # pump_max_constr2
    tupledict({(d, b, c): m.addConstr(lhs=v.C_weight[d, b, c],
                                      sense=COPT.LESS_EQUAL,
                                      rhs=data.C_pump_max[c] * data.CQ[c, 'qSPG', d] * 24 * v.b_C[d, b, c],
                                      name='C_pump_max2({0},{1},{2})'.format(d, b, c))
               for d in data.D for b in data.B for c in data.C})


def comp_storage_def(m: Model, data: ModelData, v: ModelVars):
    # comp_balance_start
    for b in data.B:
        if b == 1:
            tupledict(
                {(d, b, c): m.addConstr(lhs=v.C_stock[d, b, c] + v.deviation_supply[d, b, c],
                                        sense=COPT.EQUAL,
                                        rhs=((data.C_init[c] if d == 1 else v.C_stock[d - 1, data.B[-1], c]) +
                                             data.S[d, c] * v.blend_duration[d, b]),
                                        name='C_balance_FBD({0},{1},{2})'.format(d, b, c))
                 for d in data.D for c in data.C})
        else:
            tupledict(
                {(d, b, c): m.addConstr(lhs=v.C_stock[d, b, c] + v.deviation_supply[d, b, c],
                                        sense=COPT.EQUAL,
                                        rhs=(v.C_stock[d, b - 1, c] + data.S[d, c] * v.blend_duration[d, b]
                                             - v.C_weight[d, b, c]),
                                        name='C_balance({0},{1},{2})'.format(d, b, c))
                 for d in data.D for c in data.C})


def plan_constr_def(m: Model, data: ModelData, v: ModelVars):
    # plan_product_constr
    tupledict(
        {(d, p): m.addConstr(lhs=quicksum(v.P_weight[d, b, p] for b in data.B),
                             sense=COPT.EQUAL,
                             rhs=(data.PD_plan[d, p] if
                                  data.PD_plan[d, p] != 1e+300 else 0) - v.deviation_PD_plan_def[d, p] +
                                 v.deviation_PD_plan_exc[d, p],
                             name='plan_PD_constr({0}, {1})'.format(d, p))
         for d in data.D for p in data.P if (data.G[p] == 1 and data.PD_plan[d, p] != 0)})

    tupledict(
        {(n, p): m.addConstr(lhs=quicksum(v.P_weight[d, b, p] for b in data.B for d in data.D if data.ND[n, d] == 1),
                             sense=COPT.GREATER_EQUAL,
                             rhs=(data.PN_min[n, p] if data.PN_min[n, p] != 1e+300 else 0) - v.deviation_PN_plan[
                                 n, p],
                             name='plan_PN_min_constr({0}, {1})'.format(n, p))
         for n in data.N for p in data.P if (data.G[p] == 1 and data.PN_min[n, p] != 0)})

    tupledict(
        {(n, p): m.addConstr(lhs=quicksum(v.P_weight[d, b, p] for b in data.B for d in data.D if data.ND[n, d] == 1),
                             sense=COPT.LESS_EQUAL,
                             rhs=(data.PN_max[n, p] if data.PN_max[n, p] != 1e+300 else 0) + v.deviation_PN_plan[
                                 n, p],
                             name='plan_PN_max_constr({0}, {1})'.format(n, p))
         for n in data.N for p in data.P if (data.G[p] == 1 and data.PN_max[n, p] != 0)})


def qual_constr_def(m: Model, data: ModelData, v: ModelVars):
    p_q_min_constr = tupledict()
    p_q_max_constr = tupledict()
    for d in data.D:
        for b in data.B:
            for q in data.Q:
                for p in data.P:
                    # spec_quality_min
                    if data.PQ_min_f[p, q, d] != 0:
                        p_q_min_constr[d, p, q] = m.addConstr(
                            lhs=quicksum((v.C_feed_vol[d, b, c, p]
                                          if data.Q_vc[q] else v.C_feed[d, b, c, p]) * (data.CQ_f[c, q, d]
                                                                                        + data.Q_lb[c, q, p])
                                         for c in data.C),
                            sense=COPT.GREATER_EQUAL,
                            rhs=data.PQ_min_f[p, q, d] * (v.P_vol[d, b, p] if data.Q_vc[q] else
                                                          v.P_weight[d, b, p]) - v.deviation_PQ_spec[d, b, p, q],
                            name='PQ_min_constr({0},{1},{2},{3})'.format(d, b, p, q))

            for q in data.Q:
                for p in data.P:
                    # spec_quality_max
                    if data.PQ_max_f[p, q, d] != 0:
                        p_q_max_constr[d, p, q] = m.addConstr(
                            lhs=quicksum((v.C_feed_vol[d, b, c, p]
                                          if data.Q_vc[q] else v.C_feed[d, b, c, p]) * (data.CQ_f[c, q, d]
                                                                                        + data.Q_lb[c, q, p])
                                         for c in data.C),
                            sense=COPT.LESS_EQUAL,
                            rhs=data.PQ_max_f[p, q, d] * (v.P_vol[d, b, p] if data.Q_vc[q] else
                                                          v.P_weight[d, b, p]) + v.deviation_PQ_spec[d, b, p, q],
                            name='PQ_max_constr({0},{1},{2},{3})'.format(d, b, p, q))


def set_objective(m: Model, data: ModelData, v: ModelVars):
    # calc_penalty_supply
    m.addConstr(lhs=v.penalty_supply,
                sense=COPT.EQUAL,
                rhs=1E3 * quicksum(v.deviation_supply[d, b, c] for d in data.D for b in data.B for c in data.C),
                name='penalty_supply')

    # calc_penalty_product
    m.addConstr(lhs=v.penalty_PD_plan,
                sense=COPT.EQUAL,
                rhs=1E2 * quicksum(v.deviation_PD_plan_def[d, p] + v.deviation_PD_plan_exc[d, p]
                                   for d in data.D for p in data.P),
                name='penalty_pp_day_constr')

    m.addConstr(lhs=v.penalty_PN_plan,
                sense=COPT.EQUAL,
                rhs=1E1 * quicksum(v.deviation_PN_plan[n, p]
                                   for n in data.N for p in data.P),
                name='penalty_pp_N_constr')

    # calc_penalty_spec
    m.addConstr(lhs=v.penalty_specification, sense=COPT.EQUAL,
                rhs=1E4 * quicksum(v.deviation_PQ_spec[d, b, p, q] for d in data.D for b in data.B
                                   for p in data.P for q in data.Q),
                name='penalty_spec')

    # calc_C_total_cost
    m.addConstr(lhs=v.C_total_cost, sense=COPT.EQUAL,
                rhs=1E-6 * quicksum(data.C_cost[c] * quicksum(v.C_weight[d, b, c] for d in data.D for b in data.B)
                                    for c in data.C),
                name='C_total_cost')

    # calc_C_total_cost
    m.addConstr(lhs=v.P_total_cost, sense=COPT.EQUAL,
                rhs=1E-6 * quicksum(data.P_cost[p] * quicksum(v.P_weight[d, b, p] for d in data.D for b in data.B)
                                    for p in data.P),
                name='P_total_cost')

    # calc_profit
    m.addConstr(lhs=v.profit, sense=COPT.EQUAL,
                rhs=v.P_total_cost - v.C_total_cost,
                name='penalty_spec')

    # calc_penalty
    m.setObjective(
        v.penalty_supply + v.penalty_PD_plan + v.penalty_PN_plan + v.penalty_specification - 1E-5 * v.profit,
        sense=COPT.MINIMIZE)


if __name__ == "__main__":
    # Создаем папку расчета
    folder_name = config['input_file'].split('.')[0]
    run_path = output_folder_check(folder_name)

    # Создаем среду COPT
    env = Envr()

    # Считываем данные из GDX
    blend_problem_data = ModelData()
    blend_problem_data.import_data_excel(config, run_path)

    # Создаем экземпляр модели
    model: Model = env.createModel(blend_problem_data.model_name)

    # Инициализируем переменные модели
    variables = ModelVars()

    # Добавляем переменные в модель
    add_variables(model, blend_problem_data, variables)

    # Задаем блоки ограничений
    gasoline_logic_def(model, blend_problem_data, variables)
    blend_logic_def(model, blend_problem_data, variables)
    pump_constr_def(model, blend_problem_data, variables)
    comp_storage_def(model, blend_problem_data, variables)
    plan_constr_def(model, blend_problem_data, variables)
    qual_constr_def(model, blend_problem_data, variables)
    set_objective(model, blend_problem_data, variables)

    # Сохраняем модель в формат lp
    # model.write(run_path + '/model.lp')

    # Блок задания параметров модели
    # Ниже приведены не все возможные для задания, но наиболее актуальные для модели параметры
    model.setParam(COPT.Param.TimeLimit, 3600)  # время расчета, сек
    model.setParam(COPT.Param.RelGap, 0)  # пробел оптимальности
    model.setParam(COPT.Param.Threads, -1)  # количество потоков для использования
    model.setParam(COPT.Param.Logging, 1)  # вывод лога оптимизации
    # m.setParam(COPT.Param.LogToConsole, 1)  # вывод лога оптимизации в консоль
    model.setParam(COPT.Param.IISMethod, 1)  # диагностика конфликтов в ограничениях
    model.setLogFile(run_path + '/copt.log')
    # Инициализируем расчет оптимизационной задачи
    model.solve()

    # При успешном нахождении оптимального решения выводим значения переменных
    with open("run_results.txt", "a") as res_file:
        if model.status == COPT.OPTIMAL:
            res_file.write(run_path + ':  Optimization was stopped with status %d' % model.status)
            res_file.write('\n')
            # allvars = model.getVars()
            # print("Variable solution:")
            # for var in allvars:
            #     if var.x > 10 ** (-4) or var.name[0] in 'p':
            #         print("{0}: {1} ".format(var.name, var.x))

            print("Objective value: {} ".format(model.objval))
            # model.write(run_path + '/result.sol')
            print("Taking solution info:")
            result_df, result_dict = get_var_values(model)
            print("Writing solution to excel:")
            sol2excel(result_df, run_path, blend_problem_data.db)
            print("Making plots to visualize schedule:")
            gantt_diagram(model, blend_problem_data, variables, result_dict)
            print("Done!")

        elif model.status == COPT.TIMEOUT:
            res_file.write(run_path + ':  Optimization was stopped with status %d' % model.status)
            res_file.write('\n')
        elif model.status != COPT.INFEASIBLE:
            # print(config['scenario_number']+':  Optimization was stopped with status %d' % m.status)
            res_file.write(run_path + ':  Optimization was stopped with status %d' % model.status)
            res_file.write('\n')
        else:
            res_file.write(run_path + ':  infeasible!')
            res_file.write('\n')
            model.computeIIS()
            model.writeIIS(run_path + '/inf.ilp')
            # allvars = model.getVars()
            # print("Variable solution:")
            # for var in allvars:
            #     print("Lower IIS " + "{0}: {1} ".format(var.name, var.getLowerIIS()))
            #     print("Upper IIS " + "{0}: {1} ".format(var.name, var.getUpperIIS()))
