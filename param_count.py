import argparse
from Training.archs.vn_arch import VN
from basicsr.utils.options import parse_options
from basicsr.archs import build_network

if __name__ == '__main__':
    
    opt, _ = parse_options(root_path='.', is_train=False)
    model = build_network(opt['network_g'])

    total_parameters = model.count_parameters()

    print(f'Total amount of parameters: {total_parameters}')