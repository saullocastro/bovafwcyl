import pickle

import numpy as np
import openmdao.api as om

from vat_buck import optim_test, objective_function


class MyGA(om.ExplicitComponent):
    """Main class"""
    def __init__(self):
        self.lobpcg_X = {1 : None, 2 : None, 3 : None}
        self.cg_x0 = {1 : None, 2 : None, 3 : None}
        self.best_individual = dict(
            objective=1e6,
            desvars=None,
            it=None,
            out=None,
            )
        self.max_layers = None
        self.design_load = None
        self.max_gen = 100
        self.pop_size = 25
        self.ny = 55
        # Geometric Parameters
        self.geo_dict = dict(
            L=0.300, # length
            R=0.15 # radius
        )
        # Material Properties, default values from CS-Z Wang et al.
        self.mat_dict = dict(
            E11=90e9,
            E22=7e9,
            nu12=0.32,
            G12=4.4e9,
            G23=1.8e9,
            plyt=0.4e-3 # ply thickness
        )
        super().__init__()


    def setup(self):
        self.individuals = []
        for layer in range(self.max_layers):
            var1 = 'layer%02d_T0' % (layer+1)
            var2 = 'layer%02d_T1' % (layer+1)
            var3 = 'layer%02d_T21' % (layer+1)
            thick = 'layer%02d_thick' % (layer+1)
            self.add_input(var1, val=1.0+3.3)
            self.add_input(var2, val=1.0+3.3)
            self.add_input(var3, val=1.0+3.3)
            self.add_discrete_input(thick, val=0)
        self.add_output('objective', val=1.0)


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        it = self.iter_count
        desvars = []
        total_thick = 0
        tow_thick = 0.4e-3
        num_layers = 0
        for layer in range(self.max_layers):
            var1 = 'layer%02d_T0' % (layer+1)
            var2 = 'layer%02d_T1' % (layer+1)
            var3 = 'layer%02d_T21' % (layer+1)
            thick = 'layer%02d_thick' % (layer+1)
            thick_flag = discrete_inputs[thick]
            total_thick += thick_flag*2*tow_thick
            if thick_flag > 0:
                desvars.append((inputs[var1][0], inputs[var2][0], inputs[var3][0]))
                num_layers += thick_flag
        data = dict(inputs=inputs,
                    outputs=outputs,
                    discrete_inputs=discrete_inputs,
                    discrete_outputs=discrete_outputs,
                    desvars=desvars,
                    it=it,
                   )

        if num_layers == 0:
            out = {}
            out['lobpcg_X'] = None
            out['cg_x0'] = None
            out['Pcr'] = 1e-6
            out['volume'] = 1e15
            out['rel_vol'] = 1e15
            out['nid_pos'] = None
            out['n1s'] = None
            out['n2s'] = None
            out['n3s'] = None
            out['n4s'] = None
            out['xlin'] = None
            out['ylin'] = None
            out['ncoords'] = None
            objective = 1e15

        else:
            out = optim_test(desvars=desvars,
                             geo_prop=self.geo_dict,
                             mat_prop=self.mat_dict,
                             ny=self.ny,
                             lobpcg_X=self.lobpcg_X[num_layers],
                             cg_x0=self.cg_x0[num_layers])
            self.lobpcg_X[num_layers] = out['lobpcg_X']
            self.cg_x0[num_layers] = out['cg_x0']
            objective = objective_function(self.design_load, out)

        data['out'] = out
        outputs['objective'] = objective
        data['objective'] = objective

        self.individuals.append(data)

        if objective < self.best_individual['objective']:
            if (abs(out['Pcr']) > 0.95 * self.design_load) and (abs(out['Pcr']) < 3.0 * self.design_load):
                self.best_individual['objective'] = objective
                self.best_individual['desvars'] = desvars
                self.best_individual['it'] = it
                self.best_individual['out'] = out

        if (it + 1) % self.pop_size == 0:
            gen = (it + 1)/self.pop_size - 1
            print('Iteration:', it, 'Gen:', gen)
            print('About best individual:')
            print('  objective:', self.best_individual['objective'])
            print('  Pcr:', self.best_individual['out']['Pcr']*0.001, 'kN')
            print('  rel_vol:', self.best_individual['out']['rel_vol'])
            print('  desvars:', self.best_individual['desvars'])
            with open('GA_%04d_kN_individuals.pickle' % int(self.design_load*0.001), 'wb') as f:
                pickle.dump(self.individuals, f)
            with open('GA_%04d_kN_best_individual.pickle' % int(self.design_load*0.001), 'wb') as f:
                pickle.dump(self.best_individual, f)


if __name__ == '__main__':
    #NOTE on Apple Macbookpro M2 chipset, running in parallel using
    #     multiprocessing.Pool led to a worse performance, because the
    #     scipy.sparse.linalg solvers can already use quite well the SMP
    design_loads = [
        50e3,
        #100e3,
        #200e3,
        #500e3,
        #1000e3,
    ]
    for design_load in design_loads:
        print('___________')
        print()
        print('DESIGN LOAD', design_load)
        print('___________')
        prob = om.Problem()
        prob.model.add_subsystem('myGA', MyGA(), promotes=['*'])
        myGA = prob.model.myGA
        myGA.max_layers = 1
        myGA.design_load = design_load

        bit_size = 6   #
        # setup the optimization
        prob.driver = om.SimpleGADriver()
        prob.driver.options['max_gen'] = myGA.max_gen
        #prob.driver.options['pop_size'] = 8*bit_size
        prob.driver.options['pop_size'] = myGA.pop_size
        # prob.driver.options['cross_bits'] = True
        prob.driver.options['elitism'] = True
        prob.driver.options['Pc'] = 0.1
        prob.driver.options['Pm'] = 0.02
        prob.driver.options['run_parallel'] = False #NOTE true is not working
        # bits: 5 means 2**1 = 2 possible values for these variables
        # bits: 5 means 2**5 = 32 possible values for these variables
        # bits: 6 means 2**6 = 64 possible values for these variables

        prob.model.add_objective('objective') # minimize
        bits = []
        for layer in range(myGA.max_layers):
            var1 = 'layer%02d_T0' % (layer+1)
            var2 = 'layer%02d_T1' % (layer+1)
            var3 = 'layer%02d_T21' % (layer+1)
            thick = 'layer%02d_thick' % (layer+1)
            prob.model.add_design_var(var1, lower=3.3, upper=87.7)
            prob.model.add_design_var(var2, lower=3.3, upper=87.7)
            prob.model.add_design_var(var3, lower=3.3, upper=87.7)
            prob.model.add_design_var(thick, lower=0, upper=1)
            #bits.append((var1, bit_size)) # making it discrete
            #bits.append((var2, bit_size)) # making it discrete
            #bits.append((var3, bit_size)) # making it discrete
            bits.append((thick, 1)) # making it discrete
        prob.driver.options['bits'] = dict(bits)
        prob.setup()
        prob.run_driver()
        print('Pcr', myGA.best_individual['out']['Pcr'], 'rel_vol', myGA.best_individual['out']['rel_vol'], )
        print('ToTal iTERATIONS:'+ str(prob.driver.iter_count))
        print('Max_layer {}, Design Load {}N'.format(myGA.max_layers, myGA.design_load))
        print('objective:', prob.get_val('objective'))
        final_out = optim_test(myGA.best_individual['desvars'])
        print('Volume: {:e}'.format(final_out['volume']), '\n Pcr:', final_out['Pcr'])
