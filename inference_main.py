import io
import logging
from pathlib import Path

import numpy as np
import soundfile
import platform

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # 一定要设置的部分
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_0.pth", help='模型路径')
    parser.add_argument('-c', '--config_path', type=str, default="configs/config.json", help='配置文件路径')
    parser.add_argument('-f', '--input_files', type=str, nargs='+', default=["raw/君の知らない物語-src.wav"], help='wav文件名列表，放在raw文件夹下')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['nen'], help='合成目标说话人名称')

    # 可选项部分
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False,
                        help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt", help='聚类模型路径，如果没有训练聚类则随便填')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='聚类方案占比，范围0-1，若没有训练聚类模型则填0即可')

    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-wf', '--wav_format', type=str, default='flac', help='音频输出格式')
    parser.add_argument('-hb', '--hubert_path', type=str, default='hubert/checkpoint_best_legacy_500.pt', help='hubert模型路径')
    parser.add_argument('-o', '--output_path', type=str, default="results", help='输出路径（文件夹）')

    args = parser.parse_args()

    svc_model = Svc(args.model_path, args.config_path,
                    args.device, args.cluster_model_path, hubert_model_path=args.hubert_path)
    input_list = args.input_files
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_tool.fill_a_to_b(trans, input_list)
    for input_file, tran in zip(input_list, trans):
        wav_path = Path(input_file)
        if not wav_path.suffix or platform.system() == "Windows":
            wav_path = infer_tool.format_wav(wav_path)
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

        for spk in spk_list:
            audio = []
            for (slice_tag, data) in audio_data:
                print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

                length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
                if slice_tag:
                    print('jump empty segment')
                    _audio = np.zeros(length)
                else:
                    # padd
                    pad_len = int(audio_sr * pad_seconds)
                    data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
                    data_with_pad = io.BytesIO()
                    soundfile.write(data_with_pad, data, audio_sr, format="wav")
                    data_with_pad.seek(0)
                    out_audio, out_sr = svc_model.infer(spk, tran, data_with_pad,
                                                        cluster_infer_ratio=cluster_infer_ratio,
                                                        auto_predict_f0=auto_predict_f0,
                                                        noice_scale=noice_scale
                                                        )
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * pad_seconds)
                    _audio = _audio[pad_len:-pad_len]

                audio.extend(list(infer_tool.pad_array(_audio, length)))
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            clean_name = Path(input_file).stem
            res_path = output_dir / f'{clean_name}_{key}_{spk}{cluster_name}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)

if __name__ == '__main__':
    main()
