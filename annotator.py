#!/usr/bin/python
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b',
        '--bag',
        dest='bag',
        type=str,
        default='/dataset/AnnotationExperiment1.bag',
        help='Path to experiment bag')

    parser.add_argument(
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Show detection/display visualization')
    parser.add_argument(
        '-o',
        '--save_path',
        dest='save_path',
        default='results/',
        help='path to save the csv file')
    parser.add_argument(
        '-T',
        '--transform',
        dest='pkl',
        default='None',
        help='path to transformation pickle file')
    parser.add_argument(
        '--delay',
        dest='offset',
        default='0.0',
        help='Delay between the recordings')
    args = parser.parse_args()
    transform_path = ''
    os.system('rosparam set /use_sim_time true')
    
    if args.pkl == 'None':
        print('transformation not given. running the create_avc_transformation first')
        basename = os.path.basename(args.bag).split('.')[0] + '-tf.pkl'
        transform_path = '/code/transforms/'+basename
        if not os.path.exists('/code/transforms'):
            os.makedirs('/code/transforms')
        os.system('cd /root/catkin_ws/src/create_avc_transformations/scripts/ && python transform.py %s %s' % (args.bag, transform_path))
        
    else:
        transform_path = args.pkl
    
    print("Runing the main proccess")

    os.system('QT_X11_NO_MITSHM=1 python3 ros_listener.py {} {} {} {} {}'.format(transform_path, args.bag,args.offset,args.display,args.save_path))
    

