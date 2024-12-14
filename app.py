import gradio as gr
from pathlib import Path
from matchms import Spectrum
from typing import List, Optional, Literal
import tempfile
import numpy as np
from simms.similarity import CudaCosineGreedy, CudaModifiedCosine
from matchms.importing import load_from_mgf
from matchms import calculate_scores
import matplotlib.pyplot as plt
import pickle

# os.system("nvidia-smi")
# print("TORCH_CUDA", torch.cuda.is_available())

def preprocess_spectra(spectra: List[Spectrum]) -> Spectrum:
    from matchms.filtering import select_by_intensity, \
        normalize_intensities, \
        select_by_relative_intensity, \
        reduce_to_number_of_peaks, \
        select_by_mz, \
        require_minimum_number_of_peaks
    
    def process_spectrum(spectrum: Spectrum) -> Optional[Spectrum]:
        """
        One of the many ways to preprocess the spectrum - we use this by default.
        """
        spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
        spectrum = normalize_intensities(spectrum)
        spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
        spectrum = reduce_to_number_of_peaks(spectrum, n_max=1024)
        return spectrum
    
    spectra = list(process_spectrum(s) for s in spectra) # Some might be None
    return spectra

def run(r_filepath:Path, q_filepath:Path,
        similarity_method: Literal['CosineGreedy','ModifiedCosine'] = 'CosineGreedy',
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        shift: float = 0,
        batch_size: int = 2048,
        n_max_peaks: int = 1024,
        match_limit: int = 2048,
        array_type: Literal['sparse','numpy'] = "numpy",
        sparse_threshold: float = .75,
        do_preprocess: bool = False,
        ):
    print('\n>>>>', r_filepath, q_filepath, array_type, '\n')
    # debug = os.getenv('CUDAMS_DEBUG') == '1'
    # if debug:
    #     r_filepath = Path('tests/data/pesticides.mgf')
    #     q_filepath = Path('tests/data/pesticides.mgf')

    assert r_filepath is not None, "Reference file is missing."
    assert q_filepath is not None, "Query file is missing."

    refs, ques = list(load_from_mgf(str(r_filepath))), list(load_from_mgf(str(q_filepath)))
    if do_preprocess:
        refs = preprocess_spectra(refs)
        ques = preprocess_spectra(ques)

    # If we have small spectra, don't make a huge batch
    if batch_size > max(len(refs), len(ques)):
         batch_size = max(len(refs), len(ques))

    
    kwargs = dict(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power, shift=shift, batch_size=batch_size, 
              n_max_peaks=n_max_peaks, match_limit=match_limit, sparse_threshold=sparse_threshold)
    
    if similarity_method == 'ModifiedCosine' and shift != 0:
        gr.Error("`ModifiedCosine` can not use shift - we will proceed as if shift is 0")
    
    if similarity_method == 'ModifiedCosine':
        kwargs.pop('shift')
        
    similarity_class = CudaCosineGreedy if similarity_method == 'CosineGreedy' else CudaModifiedCosine
        
    scores_obj = calculate_scores(
        refs, ques, 
        similarity_function=similarity_class(**kwargs),
        array_type=array_type
    )

    score_vis = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)

    scores = scores_obj.to_array()
    
    outputs = len(scores.dtype.names)
    
    fig, axs = plt.subplots(1, outputs,
                            figsize=(5*outputs, 5))
    for title, ax in zip(scores.dtype.names, axs):
        ax.imshow(scores[title])
        ax.set_title(title)

    plt.suptitle("Output values")
    plt.savefig(score_vis.name)

    score = tempfile.NamedTemporaryFile(prefix='scores-', suffix='.npz', delete=False)
    np.savez(score.name, scores=scores)

    pickle_ = tempfile.NamedTemporaryFile(prefix='scores-', suffix='.pickle', delete=False)

    Path(pickle_.name).write_bytes(pickle.dumps(scores_obj))
    return score.name,  score_vis.name, pickle_.name

with gr.Blocks() as demo:
    gr.Markdown("""
    # SimMS: A GPU-Accelerated Cosine Similarity implementation for Tandem Mass Spectrometry
     
    Calculate cosine greedy similarity matrix using CUDA. See the [main repo](https://github.com/pangeai/simms) for this project. 
    This approach is x100-x500 faster than [MatchMS](https://github.com/matchms/matchms). Upload your MGF files below, or run the sample `pesticides.mgf` files against each other.
    """)
    with gr.Row():
        refs = gr.File(label="Upload REFERENCES.mgf",
                       interactive=True,
                               value='pesticides.mgf')
        ques = gr.File(label="Upload QUERIES.mgf",
                       interactive=True, value='pesticides.mgf')
    with gr.Row():
        similarity_method = gr.Radio(['CosineGreedy', 'ModifiedCosine'], value='ModifiedCosine', type='value',
                                     info="Choose one of the supported similarity methods. Need more? Let us know in github issues."
                                     )
        tolerance = gr.Number(value=0.1, label="tolerance")
        mz_power = gr.Number(value=0.0, label="m/z power")
        intensity_power = gr.Number(value=1.0, label="intensity power")
        shift = gr.Number(value=0, label="mass shift")
    with gr.Row():
        batch_size = gr.Number(value=2048, label="Batch Size", 
                                info='Compare this many spectra to same amount of other spectra at each iteration.')
        n_max_peaks = gr.Number(value=1024, label="Maximum Number of Peaks", 
                                info="Consider this many m/z peaks at most, per spectrum.")
        match_limit = gr.Number(value=2048, label="Match Limit", 
                                info="Consider this many pairs of m/z before stopping. "
                                    "In practice, a value of 2048 gives more than 99.99% accuracy on GNPS")
        do_preprocess = gr.Checkbox(value=False, label="filter spectra", 
                                    info="If you want to filter spectra before processing, we can do that. Look at the code to see details.")
    with gr.Row():
        array_type = gr.Radio(['numpy', 'sparse'], 
                              value='numpy', type='value',
                              label='If `sparse`, everything with score less than `sparse_threshold` will be discarded.'
                                    'If `numpy`, we disable sparse behaviour.')
        sparse_threshold = gr.Slider(minimum=0, maximum=1, value=0.75, label="Sparse Threshold",
                                        info="For very large results, when comparing, more than 10k x 10k, the output dense score matrix can grow too large for RAM."
                                        "While most of the scores aren't useful (near zero). This argument discards all scores less than sparse_threshold, and returns "
                                        "results as a SparseStack format."
                                        )
    with gr.Row():
        score_vis = gr.Image()

    with gr.Row():
        out_npz = gr.File(label="Download similarity matrix (.npz format)", 
                      interactive=False)
        out_pickle = gr.File(label="Download full `Scores` object (.pickle format)", 
                      interactive=False)
    gr.Markdown("""
            **NOTE** You can use this snippet to use the downloaded array:
            ```py
            import numpy as np
            arr = np.load('scores-nr0hqp85.npz')['scores']
            print(arr)
            ```""")
    btn = gr.Button("Run")
    btn.click(fn=run, 
            inputs=[refs, ques, similarity_method, tolerance, mz_power, intensity_power, shift, 
                            batch_size, n_max_peaks, match_limit, 
                            array_type, sparse_threshold, do_preprocess], 
            outputs=[out_npz, score_vis, out_pickle])

if __name__ == "__main__":
    demo.launch(debug=True)