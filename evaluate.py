import os, sys, subprocess, argparse, git, re, yaml, zipfile, shutil
from subprocess import Popen
from git import Repo
from pathlib import Path
from tqdm import tqdm
from yolov8.ultralytics.yolo.utils import LOGGER
from yolov8.ultralytics.yolo.utils.checks import check_requirements, print_args
from yolov8.ultralytics.yolo.utils.files import increment_path
from torch.utils.tensorboard import SummaryWriter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
MODELS = ROOT / 'weights'

# Add directories to PATH.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort')) # Necessary.

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class Evaluator:
    def __init__(self, opts):
        self.opt = opts

    def download_mot_eval_tools(self, eval_tools_path):
        eval_tools_repo = "https://github.com/JonathonLuiten/TrackEval"
        try:
            Repo.clone_from(eval_tools_repo, eval_tools_path)
            LOGGER.info('Evaluation repository downloaded.')
        except git.exc.GitError as err:
            LOGGER.info('Evaluation repository already downloaded.')

    def download_mot_dataset(self, eval_tools_path, benchmark):
        url = 'https://motchallenge.net/data/' + benchmark + '.zip'
        zip_dst = eval_tools_path / (benchmark + '.zip')
        if not (eval_tools_path / 'data' / benchmark).exists():
            os.system(f"curl -# -L {url} -o {zip_dst} -# --retry 3 -C -")
            LOGGER.info(f'{benchmark}.zip downloaded sucessfully')

            try:
                with zipfile.ZipFile((eval_tools_path / (benchmark + '.zip')), 'r') as zip_file:

                    # MOT16 uses a different structure.
                    if self.opt.benchmark == 'MOT16':
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            member_path = eval_tools_path / 'data' / 'MOT16' / member
                            if not member_path.exists() and not member_path.is_file():
                                zip_file.extract(member, eval_tools_path / 'data' / 'MOT16')
                    else:
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            member_path = eval_tools_path / 'data' / member
                            if not member_path.exists() and not member_path.is_file():
                                zip_file.extract(member, eval_tools_path / 'data')

                LOGGER.info(f'{benchmark}.zip unzipped successfully.')

            except Exception:
                print(f'{benchmark}.zip is broken.')
                sys.exit()

    def eval_setup(self, opt, eval_tools_path):
        gt_folder = eval_tools_path / 'data' / self.opt.benchmark / self.opt.split
        mot_seqs_path = eval_tools_path / 'data' / opt.benchmark / opt.split
        if opt.benchmark == 'MOT17':
            seq_paths = sorted([str(p / 'img1') for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()])
            seq_paths = [Path(p) for p in seq_paths if 'FRCNN' in p]
        elif opt.benchmark == 'MOT16' or opt.benchmark == 'MOT20':
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]
        elif opt.benchmark == 'MOT17-mini':
            mot_seqs_path = Path('./eval_tools/data') / self.opt.benchmark / self.opt.split
            gt_folder = Path('./eval_tools/data') / self.opt.benchmark / self.opt.split
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]

        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # Increment run.
        MOT_results_folder = eval_tools_path / 'data' / 'trackers' / 'mot_challenge' / opt.benchmark / save_dir.name / 'data'
        (MOT_results_folder).mkdir(parents=True, exist_ok=True)
        return seq_paths, save_dir, MOT_results_folder, gt_folder

    def device_setup(self, opt, seq_paths):

        # Extend devices to as many sequences are available.
        if any(isinstance(i, int) for i in opt.device) and len(opt.device) > 1:
            devices = opt.device
            for a in range(0, len(opt.device) % len(seq_paths)):
                opt.device.extend(devices)
            opt.device = opt.device[:len(seq_paths)]
        free_devices = opt.device * opt.threads
        return free_devices

    def eval(self, opt, seq_paths, save_dir, MOT_results_folder, eval_tools_path, gt_folder, free_devices):

        if not self.opt.eval_existing:
            processes = []

            busy_devices = []
            for i, seq_path in enumerate(seq_paths):
                if i > 0 and len(free_devices) == 0:
                    if len(processes) == 0:
                        raise IndexError("No processes and no devices.")

                    processes.pop(0).wait()
                    free_devices.append(busy_devices.pop(0))

                tracking_subprocess_device = free_devices.pop(0)
                busy_devices.append(tracking_subprocess_device)

                dst_seq_path = seq_path.parent / seq_path.parent.name

                if not dst_seq_path.is_dir():
                    src_seq_path = seq_path
                    shutil.move(str(src_seq_path), str(dst_seq_path))

                p = subprocess.Popen([
                    sys.executable, "track.py",
                    "--detector", self.opt.detector,
                    "--re-id", self.opt.re_id,
                    "--tracker", self.opt.tracker,
                    "--conf-thres", str(self.opt.conf_thres),
                    "--imgsz", str(self.opt.imgsz[0]),
                    "--classes", str(0),
                    "--name", save_dir.name,
                    "--project", self.opt.project,
                    "--device", str(tracking_subprocess_device),
                    "--source", dst_seq_path,
                    "--exist-ok",
                    "--save-txt",
                ])
                processes.append(p)

            for p in processes:
                p.wait()

        print_args(vars(self.opt))

        results = (save_dir.parent / self.opt.eval_existing / 'tracks' if self.opt.eval_existing else save_dir / 'tracks').glob('*.txt')
        for src in results:
            if self.opt.eval_existing:
                dst = MOT_results_folder.parent.parent / self.opt.eval_existing / 'data' / Path(src.stem + '.txt')
            else:
                dst = MOT_results_folder / Path(src.stem + '.txt')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

        # Run the evaluation on the generated txts.
        d = [seq_path.parent.name for seq_path in seq_paths]
        p = subprocess.run(
            args=[
                sys.executable, eval_tools_path / 'scripts' / 'run_mot_challenge.py',
                "--GT_FOLDER", gt_folder,
                "--BENCHMARK", self.opt.benchmark,
                "--TRACKERS_TO_EVAL", self.opt.eval_existing if self.opt.eval_existing else self.opt.benchmark,
                "--SPLIT_TO_EVAL", "train",
                "--METRICS", "HOTA", "CLEAR", "Identity",
                "--USE_PARALLEL", "True",
                "--TRACKER_SUB_FOLDER", str(Path(*Path(MOT_results_folder).parts[-2:])),
                "--NUM_PARALLEL_CORES", "4",
                "--SKIP_SPLIT_FOL", "True",
                "--SEQ_INFO"
            ] + d, universal_newlines=True, stdout=subprocess.PIPE
        )

        print(p.stdout)

        with open(save_dir / 'MOT_results.txt', 'w') as f:
            f.write(p.stdout)

        shutil.copyfile(opt.track_config, save_dir / opt.track_config.name)
        return p.stdout

    def parse_mot_results(self, results):
        all_results = results.split('COMBINED')[2:-1]
        all_results = [float(re.findall("[-+]?(?:\d*\.*\d+)", f)[0]) for f in all_results]
        all_results = {key: value for key, value in zip(['HOTA', 'MOTA', 'IDF1'], all_results)}
        return all_results

    def run(self, opt):
        evaluation = Evaluator(opt)
        eval_tools_path = ROOT / 'eval_tools'
        evaluation.download_mot_eval_tools(eval_tools_path)

        if any(opt.benchmark == s for s in ['MOT16', 'MOT17', 'MOT20']):
            evaluation.download_mot_dataset(eval_tools_path, opt.benchmark)

        seq_paths, save_dir, MOT_results_folder, gt_folder = evaluation.eval_setup(opt, eval_tools_path)
        free_devices = evaluation.device_setup(opt, seq_paths)
        results = evaluation.eval(opt, seq_paths, save_dir, MOT_results_folder, eval_tools_path, gt_folder, free_devices)
        all_results = self.parse_mot_results(results)

        # Add to tensorboard.
        writer = SummaryWriter(save_dir)
        writer.add_scalar('HOTA', all_results['HOTA'])
        writer.add_scalar('MOTA', all_results['MOTA'])
        writer.add_scalar('IDF1', all_results['IDF1'])

        return all_results

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, default=MODELS / 'yolov8n.pt')
    parser.add_argument('--re-id', type=str, default=MODELS / 'lmbn_n_cuhk03_d.pt')
    parser.add_argument('--tracker', type=str, default='botsort')
    parser.add_argument('--track-config', type=Path, default=None)
    parser.add_argument('--name', default='run')
    parser.add_argument('--project', default=ROOT / 'results' / 'evaluations')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--benchmark', type=str, default='MOT17-mini')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--eval-existing', type=str, default='')
    parser.add_argument('--conf-thres', type=float, default=0.45)
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280])
    parser.add_argument('--device', default='')
    parser.add_argument('--threads', type=int, default=1)

    opt = parser.parse_args()
    opt.track_config = ROOT / 'trackers' / opt.tracker / 'configs' / (opt.tracker + '.yaml')
    with open(opt.track_config, 'r') as f:
        params = yaml.load(f, Loader=yaml.loader.SafeLoader)
        opt.conf_thres = params[opt.tracker]['conf_thres']

    device = []

    for a in opt.device.split(','):
        try:
            a = int(a)
        except ValueError:
            pass
        device.append(a)
    opt.device = device

    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    check_requirements(requirements=ROOT / 'requirements.txt')
    evaluation = Evaluator(opt)
    evaluation.run(opt)
