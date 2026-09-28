// Vanilla JS frontend for Taiwanese band chart converter
document.addEventListener("DOMContentLoaded", () => {
  // State
  let selectedFiles = [];
  let currentSheetId = null;
  let currentParsedSheet = null;
  let pollInterval = null;

  // DOM Elements - Steps & Navigation
  const stepNavs = [1, 2, 3, 4].map(n => document.getElementById(`step-nav-${n}`));
  const stepSections = [1, 2, 3, 4].map(n => document.getElementById(`step-${n}`));

  // Step 1 Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const thumbContainer = document.getElementById("thumb-container");
  const thumbGrid = document.getElementById("thumb-grid");
  const btnAddMore = document.getElementById("btn-add-more");
  const btnStartUpload = document.getElementById("btn-start-upload");

  // Step 2 Elements
  const parsingStatusTitle = document.getElementById("parsing-status-title");
  const parsingStatusDesc = document.getElementById("parsing-status-desc");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const parsingErrorBox = document.getElementById("parsing-error-box");
  const parsingErrorMsg = document.getElementById("parsing-error-msg");
  const btnRetryUpload = document.getElementById("btn-retry-upload");

  // Step 3 Elements
  const hdrTitle = document.getElementById("hdr-title");
  const hdrStyle = document.getElementById("hdr-style");
  const hdrTempo = document.getElementById("hdr-tempo");
  const hdrTimesig = document.getElementById("hdr-timesig");
  const printedKeysWrapper = document.getElementById("printed-keys-wrapper");
  const selectStartKey = document.getElementById("select-start-key");
  const selectInstrument = document.getElementById("select-instrument");
  const systemsContainer = document.getElementById("systems-container");
  const btnSaveParsed = document.getElementById("btn-save-parsed");
  const btnTriggerRender = document.getElementById("btn-trigger-render");
  const btnTriggerRenderBottom = document.getElementById("btn-trigger-render-bottom");

  // Step 4 Elements
  const btnBackToEdit = document.getElementById("btn-back-to-edit");
  const btnDownloadPdf = document.getElementById("btn-download-pdf");
  const renderPreviews = document.getElementById("render-previews");
  const reSelectKey = document.getElementById("re-select-key");
  const reSelectDifficulty = document.getElementById("re-select-difficulty");
  const btnReRender = document.getElementById("btn-re-render");

  // Switch Active Step
  function setStep(stepNum) {
    stepNavs.forEach((nav, idx) => {
      const n = idx + 1;
      nav.classList.remove("active", "completed");
      if (n === stepNum) {
        nav.classList.add("active");
      } else if (n < stepNum) {
        nav.classList.add("completed");
      }
    });

    stepSections.forEach((sec, idx) => {
      if (idx + 1 === stepNum) {
        sec.classList.remove("hidden");
      } else {
        sec.classList.add("hidden");
      }
    });
  }

  // --- Step 1: File Selection & Thumbnails ---
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    handleNewFiles(e.dataTransfer.files);
  });

  fileInput.addEventListener("change", (e) => {
    handleNewFiles(e.target.files);
    fileInput.value = "";
  });

  btnAddMore.addEventListener("click", () => fileInput.click());

  function handleNewFiles(fileList) {
    if (!fileList || fileList.length === 0) return;

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      if (selectedFiles.length >= 12) {
        alert("最多仅支持上传 12 个文件！");
        break;
      }
      if (file.size > 20 * 1024 * 1024) {
        alert(`文件 ${file.name} 超过 20MB 限制！`);
        continue;
      }
      const ext = file.name.split(".").pop().toLowerCase();
      if (!["jpg", "jpeg", "png", "webp", "pdf"].includes(ext)) {
        alert(`文件 ${file.name} 格式不受支持（仅支持 JPG, PNG, WEBP, PDF）`);
        continue;
      }
      selectedFiles.push(file);
    }

    renderThumbnails();
  }

  function renderThumbnails() {
    thumbGrid.innerHTML = "";
    if (selectedFiles.length === 0) {
      thumbContainer.classList.add("hidden");
      btnStartUpload.disabled = true;
      return;
    }

    thumbContainer.classList.remove("hidden");
    btnStartUpload.disabled = false;

    selectedFiles.forEach((file, index) => {
      const card = document.createElement("div");
      card.className = "thumb-card";

      const isPdf = file.name.toLowerCase().endsWith(".pdf");
      if (isPdf) {
        const pdfIcon = document.createElement("div");
        pdfIcon.className = "thumb-pdf-icon";
        pdfIcon.textContent = "PDF";
        card.appendChild(pdfIcon);
      } else {
        const img = document.createElement("img");
        img.className = "thumb-img";
        img.src = URL.createObjectURL(file);
        card.appendChild(img);
      }

      const title = document.createElement("div");
      title.className = "thumb-title";
      title.title = file.name;
      title.textContent = `${index + 1}. ${file.name}`;
      card.appendChild(title);

      const actions = document.createElement("div");
      actions.className = "thumb-actions";

      if (index > 0) {
        const btnUp = document.createElement("button");
        btnUp.type = "button";
        btnUp.className = "btn btn-secondary btn-sm";
        btnUp.textContent = "←";
        btnUp.title = "前移";
        btnUp.onclick = () => {
          const tmp = selectedFiles[index - 1];
          selectedFiles[index - 1] = selectedFiles[index];
          selectedFiles[index] = tmp;
          renderThumbnails();
        };
        actions.appendChild(btnUp);
      }

      if (index < selectedFiles.length - 1) {
        const btnDown = document.createElement("button");
        btnDown.type = "button";
        btnDown.className = "btn btn-secondary btn-sm";
        btnDown.textContent = "→";
        btnDown.title = "后移";
        btnDown.onclick = () => {
          const tmp = selectedFiles[index + 1];
          selectedFiles[index + 1] = selectedFiles[index];
          selectedFiles[index] = tmp;
          renderThumbnails();
        };
        actions.appendChild(btnDown);
      }

      const btnDel = document.createElement("button");
      btnDel.type = "button";
      btnDel.className = "btn btn-danger btn-sm";
      btnDel.textContent = "✕";
      btnDel.title = "移除";
      btnDel.onclick = () => {
        selectedFiles.splice(index, 1);
        renderThumbnails();
      };
      actions.appendChild(btnDel);

      card.appendChild(actions);
      thumbGrid.appendChild(card);
    });
  }

  // --- Step 2: Upload & Poll ---
  btnStartUpload.addEventListener("click", async () => {
    if (selectedFiles.length === 0) return;

    setStep(2);
    parsingStatusTitle.textContent = "正在上传文件...";
    parsingStatusDesc.textContent = "正在将乐谱图片/PDF传输至服务端并切片...";
    progressBarFill.style.width = "10%";
    parsingErrorBox.classList.add("hidden");

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const resp = await fetch("/api/sheets", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `上传失败 (HTTP ${resp.status})`);
      }

      const data = await resp.json();
      currentSheetId = data.sheet_id;
      startPolling(currentSheetId);
    } catch (err) {
      showParsingError(err.message);
    }
  });

  btnRetryUpload.addEventListener("click", () => {
    if (pollInterval) clearInterval(pollInterval);
    setStep(1);
  });

  function showParsingError(msg) {
    if (pollInterval) clearInterval(pollInterval);
    parsingErrorMsg.textContent = `解析出错: ${msg}`;
    parsingErrorBox.classList.remove("hidden");
    parsingStatusTitle.textContent = "解析未完成";
  }

  function startPolling(sheetId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const resp = await fetch(`/api/sheets/${sheetId}`);
        if (!resp.ok) {
          throw new Error(`获取状态失败 (HTTP ${resp.status})`);
        }
        const state = await resp.json();

        if (state.status === "parsing") {
          parsingStatusTitle.textContent = "正在进行智能乐谱识别...";
          parsingStatusDesc.textContent = "视觉模型正在识别小节线、简谱旋律、调号与级数和弦...";
          const pct = Math.min(Math.round((state.progress || 0.3) * 100), 90);
          progressBarFill.style.width = `${pct}%`;
        } else if (state.status === "ready") {
          clearInterval(pollInterval);
          progressBarFill.style.width = "100%";
          currentParsedSheet = state.parsed;
          renderStep3();
          setStep(3);
        } else if (state.status === "error") {
          showParsingError(state.error || "未知解析错误");
        }
      } catch (err) {
        showParsingError(err.message);
      }
    }, 2000);
  }

  // --- Step 3: Review & Edit ---
  function renderStep3() {
    if (!currentParsedSheet) return;

    const hdr = currentParsedSheet.header || {};
    hdrTitle.value = hdr.title || "";
    hdrStyle.value = hdr.style || "";
    hdrTempo.value = hdr.tempo_bpm || "";
    hdrTimesig.value = hdr.time_signature || "4/4";

    // Printed Key Badges
    printedKeysWrapper.innerHTML = "";
    const keyBadges = [
      { label: "原调", key: hdr.original_key },
      { label: "男调", key: hdr.male_key },
      { label: "女调", key: hdr.female_key },
    ];

    keyBadges.forEach(({ label, key }) => {
      if (key) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "printed-key-btn";
        btn.textContent = `${label}: ${key}`;
        btn.onclick = () => {
          selectStartKey.value = key;
        };
        printedKeysWrapper.appendChild(btn);
      }
    });

    if (hdr.original_key) {
      // Find matching option or set value
      const hasOpt = Array.from(selectStartKey.options).some(o => o.value === hdr.original_key);
      if (hasOpt) selectStartKey.value = hdr.original_key;
    }

    renderSystemsAndMeasures();
  }

  function renderSystemsAndMeasures() {
    systemsContainer.innerHTML = "";
    const systems = currentParsedSheet.systems || [];
    const keyChanges = currentParsedSheet.key_changes || [];

    systems.forEach((sys, sysIdx) => {
      const card = document.createElement("div");
      card.className = "system-card";

      const header = document.createElement("div");
      header.className = "system-header";
      header.innerHTML = `
        <span>第 ${sys.page + 1} 页 · 行 #${sysIdx + 1} ${sys.section_label ? `(${sys.section_label})` : ""}</span>
        <span style="font-size: 0.75rem; color: var(--text-muted);">${sys.measures ? sys.measures.length : 0} 个小节</span>
      `;
      card.appendChild(header);

      const measuresRow = document.createElement("div");
      measuresRow.className = "measures-row";

      (sys.measures || []).forEach((m) => {
        const mBox = document.createElement("div");
        mBox.className = "measure-box";

        const mNum = document.createElement("div");
        mNum.className = "measure-num";
        mNum.innerHTML = `<span>小节 #${m.index + 1}</span> <span>${m.beats || 4} 拍</span>`;
        mBox.appendChild(mNum);

        if (m.melody) {
          const melodyDiv = document.createElement("div");
          melodyDiv.style.fontSize = "0.8rem";
          melodyDiv.style.color = "var(--primary)";
          melodyDiv.style.fontWeight = "600";
          melodyDiv.textContent = `旋律: ${m.melody}`;
          mBox.appendChild(melodyDiv);
        }

        // Chords List
        const chordList = document.createElement("div");
        chordList.className = "chord-list";

        (m.chords || []).forEach((c, cIdx) => {
          const cItem = document.createElement("div");
          cItem.className = "chord-item";

          const rawInput = document.createElement("input");
          rawInput.type = "text";
          rawInput.className = "chord-input";
          rawInput.value = c.raw || "";
          rawInput.placeholder = "和弦";
          rawInput.onchange = (e) => {
            c.raw = e.target.value;
          };

          const beatInput = document.createElement("input");
          beatInput.type = "number";
          beatInput.step = "0.5";
          beatInput.className = "beat-input";
          beatInput.value = c.beat || 1;
          beatInput.placeholder = "拍";
          beatInput.onchange = (e) => {
            c.beat = parseFloat(e.target.value) || 1.0;
          };

          const delBtn = document.createElement("button");
          delBtn.type = "button";
          delBtn.className = "btn btn-danger btn-sm";
          delBtn.textContent = "✕";
          delBtn.style.padding = "2px 6px";
          delBtn.onclick = () => {
            m.chords.splice(cIdx, 1);
            renderSystemsAndMeasures();
          };

          cItem.appendChild(rawInput);
          cItem.appendChild(beatInput);
          cItem.appendChild(delBtn);
          chordList.appendChild(cItem);
        });

        mBox.appendChild(chordList);

        // Add Chord Button
        const addChordBtn = document.createElement("button");
        addChordBtn.type = "button";
        addChordBtn.className = "btn btn-secondary btn-sm";
        addChordBtn.style.marginTop = "4px";
        addChordBtn.textContent = "+ 和弦";
        addChordBtn.onclick = () => {
          if (!m.chords) m.chords = [];
          m.chords.push({ raw: "1", beat: 1.0, bbox: null });
          renderSystemsAndMeasures();
        };
        mBox.appendChild(addChordBtn);

        // Key change notice if applicable
        const kc = keyChanges.find(k => k.at_measure === m.index);
        if (kc) {
          const kcBadge = document.createElement("div");
          kcBadge.className = "key-change-badge";
          kcBadge.innerHTML = `<span>转调: ${kc.raw} (${kc.semitones > 0 ? "+" : ""}${kc.semitones})</span>`;
          mBox.appendChild(kcBadge);
        }

        measuresRow.appendChild(mBox);
      });

      card.appendChild(measuresRow);
      systemsContainer.appendChild(card);
    });
  }

  function collectFormDataToParsedSheet() {
    if (!currentParsedSheet) return;
    if (!currentParsedSheet.header) currentParsedSheet.header = {};
    currentParsedSheet.header.title = hdrTitle.value.trim();
    currentParsedSheet.header.style = hdrStyle.value.trim();
    currentParsedSheet.header.tempo_bpm = hdrTempo.value ? parseFloat(hdrTempo.value) : null;
    currentParsedSheet.header.time_signature = hdrTimesig.value.trim() || "4/4";
  }

  async function saveParsedSheet() {
    collectFormDataToParsedSheet();
    try {
      const resp = await fetch(`/api/sheets/${currentSheetId}/parsed`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentParsedSheet),
      });
      if (!resp.ok) {
        throw new Error(`保存失败 (HTTP ${resp.status})`);
      }
      return true;
    } catch (err) {
      alert(`保存修改失败: ${err.message}`);
      return false;
    }
  }

  btnSaveParsed.addEventListener("click", async () => {
    btnSaveParsed.disabled = true;
    const ok = await saveParsedSheet();
    btnSaveParsed.disabled = false;
    if (ok) {
      alert("乐谱修改已保存成功！");
    }
  });

  // --- Step 4: Render ---
  async function triggerRender(key, diff, inst) {
    const ok = await saveParsedSheet();
    if (!ok) return;

    const btn = btnTriggerRender;
    const bottomBtn = btnTriggerRenderBottom;
    btn.disabled = true;
    bottomBtn.disabled = true;
    btn.textContent = "正在生成伴奏...";
    bottomBtn.textContent = "正在生成伴奏...";

    try {
      const resp = await fetch(`/api/sheets/${currentSheetId}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_key: key,
          difficulty: diff,
          instrument: inst,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `生成伴奏失败 (HTTP ${resp.status})`);
      }

      const result = await resp.json();
      displayRenderResult(result, key, diff);
      setStep(4);
    } catch (err) {
      alert(`伴奏生成失败: ${err.message}`);
    } finally {
      btn.disabled = false;
      bottomBtn.disabled = false;
      btn.textContent = "生成伴奏谱 🎶";
      bottomBtn.textContent = "生成伴奏谱 🎶";
    }
  }

  function getSelectedDifficulty() {
    const radios = document.getElementsByName("difficulty");
    for (let r of radios) {
      if (r.checked) return r.value;
    }
    return "intermediate";
  }

  btnTriggerRender.addEventListener("click", () => {
    triggerRender(selectStartKey.value, getSelectedDifficulty(), selectInstrument.value);
  });

  btnTriggerRenderBottom.addEventListener("click", () => {
    triggerRender(selectStartKey.value, getSelectedDifficulty(), selectInstrument.value);
  });

  function displayRenderResult(result, key, diff) {
    btnDownloadPdf.href = result.pdf_url;
    reSelectKey.value = key;
    reSelectDifficulty.value = diff;

    renderPreviews.innerHTML = "";
    (result.preview_urls || []).forEach((url, idx) => {
      const card = document.createElement("div");
      card.className = "preview-page-card";

      const header = document.createElement("div");
      header.className = "preview-page-header";
      header.textContent = `第 ${idx + 1} 页伴奏预览`;
      card.appendChild(header);

      const img = document.createElement("img");
      img.className = "preview-page-img";
      img.src = url;
      img.alt = `第 ${idx + 1} 页伴奏`;
      img.loading = "lazy";
      card.appendChild(img);

      renderPreviews.appendChild(card);
    });
  }

  btnReRender.addEventListener("click", async () => {
    btnReRender.disabled = true;
    btnReRender.textContent = "生成中...";
    try {
      await triggerRender(reSelectKey.value, reSelectDifficulty.value, selectInstrument.value);
    } finally {
      btnReRender.disabled = false;
      btnReRender.textContent = "重新生成";
    }
  });

  btnBackToEdit.addEventListener("click", () => {
    setStep(3);
  });
});
