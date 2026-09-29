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

  // Step 3 QA Elements
  const qaSummaryBar = document.getElementById("qa-summary-bar");
  const qaSummaryText = document.getElementById("qa-summary-text");
  const qaLayoutGateBanner = document.getElementById("qa-layout-gate-banner");
  const qaLayoutGateDesc = document.getElementById("qa-layout-gate-desc");
  const qaLayoutGateDetails = document.getElementById("qa-layout-gate-details");
  const renderErrorAlert = document.getElementById("render-error-alert");
  const renderErrorMessage = document.getElementById("render-error-message");
  const qaNeedsReviewSection = document.getElementById("qa-needs-review-section");
  const qaNeedsReviewList = document.getElementById("qa-needs-review-list");
  const qaAutofixedDetails = document.getElementById("qa-autofixed-details");
  const qaAutofixedSummary = document.getElementById("qa-autofixed-summary");
  const qaAutofixedList = document.getElementById("qa-autofixed-list");
  const measuresDetails = document.getElementById("measures-details");

  function showRenderError(msg) {
    if (renderErrorAlert && renderErrorMessage) {
      renderErrorMessage.textContent = msg;
      renderErrorAlert.classList.remove("hidden");
    }
  }

  function hideRenderError() {
    if (renderErrorAlert) {
      renderErrorAlert.classList.add("hidden");
    }
  }

  // Lightbox Elements
  const lightboxModal = document.getElementById("lightbox-modal");
  const lightboxBackdrop = document.getElementById("lightbox-backdrop");
  const lightboxTitle = document.getElementById("lightbox-title");
  const lightboxClose = document.getElementById("lightbox-close");
  const lightboxZoomIn = document.getElementById("lightbox-zoom-in");
  const lightboxZoomOut = document.getElementById("lightbox-zoom-out");
  const lightboxZoomReset = document.getElementById("lightbox-zoom-reset");
  const lightboxZoomLevel = document.getElementById("lightbox-zoom-level");
  const lightboxViewport = document.getElementById("lightbox-viewport");
  const lightboxCanvasContainer = document.getElementById("lightbox-canvas-container");
  const lightboxCanvas = document.getElementById("lightbox-canvas");

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

  // --- Lightbox Implementation (Zoom & Pan) ---
  const lbState = {
    scale: 1.0,
    translateX: 0,
    translateY: 0,
    isDragging: false,
    startX: 0,
    startY: 0,
  };

  function updateLightboxTransform() {
    if (!lightboxCanvasContainer) return;
    lightboxCanvasContainer.style.transform = `translate(${lbState.translateX}px, ${lbState.translateY}px) scale(${lbState.scale})`;
    if (lightboxZoomLevel) {
      lightboxZoomLevel.textContent = `${Math.round(lbState.scale * 100)}%`;
    }
  }

  function resetLightboxTransform() {
    lbState.scale = 1.0;
    lbState.translateX = 0;
    lbState.translateY = 0;
    updateLightboxTransform();
  }

  function openLightbox(title, drawFn) {
    if (!lightboxModal) return;
    if (lightboxTitle) lightboxTitle.textContent = title || "🔍 乐谱高清大图";
    lightboxModal.classList.remove("hidden");
    resetLightboxTransform();
    if (drawFn && lightboxCanvas) {
      drawFn(lightboxCanvas);
    }
  }

  function closeLightbox() {
    if (lightboxModal) lightboxModal.classList.add("hidden");
  }

  if (lightboxClose) lightboxClose.onclick = closeLightbox;
  if (lightboxBackdrop) lightboxBackdrop.onclick = closeLightbox;
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightboxModal && !lightboxModal.classList.contains("hidden")) {
      closeLightbox();
    }
  });

  if (lightboxZoomIn) {
    lightboxZoomIn.onclick = () => {
      lbState.scale = Math.min(lbState.scale * 1.25, 6.0);
      updateLightboxTransform();
    };
  }
  if (lightboxZoomOut) {
    lightboxZoomOut.onclick = () => {
      lbState.scale = Math.max(lbState.scale / 1.25, 0.25);
      updateLightboxTransform();
    };
  }
  if (lightboxZoomReset) {
    lightboxZoomReset.onclick = resetLightboxTransform;
  }

  if (lightboxViewport) {
    lightboxViewport.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
      lbState.scale = Math.min(Math.max(lbState.scale * zoomFactor, 0.25), 6.0);
      updateLightboxTransform();
    }, { passive: false });

    lightboxViewport.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      lbState.isDragging = true;
      lbState.startX = e.clientX - lbState.translateX;
      lbState.startY = e.clientY - lbState.translateY;
      lightboxViewport.classList.add("dragging");
    });

    window.addEventListener("mousemove", (e) => {
      if (!lbState.isDragging) return;
      lbState.translateX = e.clientX - lbState.startX;
      lbState.translateY = e.clientY - lbState.startY;
      updateLightboxTransform();
    });

    window.addEventListener("mouseup", () => {
      if (lbState.isDragging) {
        lbState.isDragging = false;
        if (lightboxViewport) lightboxViewport.classList.remove("dragging");
      }
    });

    let lastTouchDist = 0;
    lightboxViewport.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) {
        lbState.isDragging = true;
        lbState.startX = e.touches[0].clientX - lbState.translateX;
        lbState.startY = e.touches[0].clientY - lbState.translateY;
      } else if (e.touches.length === 2) {
        lbState.isDragging = false;
        lastTouchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    }, { passive: true });

    lightboxViewport.addEventListener("touchmove", (e) => {
      if (e.touches.length === 1 && lbState.isDragging) {
        lbState.translateX = e.touches[0].clientX - lbState.startX;
        lbState.translateY = e.touches[0].clientY - lbState.startY;
        updateLightboxTransform();
      } else if (e.touches.length === 2) {
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        if (lastTouchDist > 0) {
          const factor = dist / lastTouchDist;
          lbState.scale = Math.min(Math.max(lbState.scale * factor, 0.25), 6.0);
          updateLightboxTransform();
        }
        lastTouchDist = dist;
      }
    }, { passive: true });

    lightboxViewport.addEventListener("touchend", () => {
      lbState.isDragging = false;
      lastTouchDist = 0;
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

  // --- QA Review Card Builders & Helpers ---
  function isStructuralIssue(issue) {
    const code = issue.code || "";
    return code === "barline_count_mismatch" || code.startsWith("barline_") || code.startsWith("structural_") || code.includes("measure_count");
  }

  function getIssueType(issue) {
    const code = issue.code || "";
    if (isStructuralIssue(issue)) return "structural";
    if (code.startsWith("key_change")) return "key_change";
    if (code.startsWith("melody_") || code.includes("beat_sum")) return "melody_beat";
    if (code.startsWith("missing_chord")) return "missing_chord";
    return "chord";
  }

  async function confirmIssue(payload) {
    try {
      const resp = await fetch(`/api/sheets/${currentSheetId}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `操作失败 (HTTP ${resp.status})`);
      }
      const data = await resp.json();
      if (data && data.parsed) {
        currentParsedSheet = data.parsed;
        renderStep3();
      }
      return true;
    } catch (err) {
      alert(`核对提交失败: ${err.message}`);
      return false;
    }
  }

  function buildStructuralCard(issue, issueIdx) {
    const card = document.createElement("div");
    card.className = "qa-review-card structural";

    let targetSystem = null;
    let targetSystemIndex = 0;
    if (issue.measure_index !== null && issue.measure_index !== undefined) {
      (currentParsedSheet.systems || []).forEach((sys, idx) => {
        if (sys.measures && sys.measures.some(m => m.index === issue.measure_index)) {
          targetSystem = sys;
          targetSystemIndex = idx;
        }
      });
    }
    if (!targetSystem) {
      const mMatch = /第\s*(\d+)\s*页第\s*(\d+)\s*行/.exec(issue.message) || /Page\s*(\d+)\s*System\s*(\d+)/.exec(issue.detail ? issue.detail.warning || "" : "");
      if (mMatch) {
        const p = parseInt(mMatch[1], 10) - 1;
        const s = parseInt(mMatch[2], 10) - 1;
        const pSys = (currentParsedSheet.systems || []).filter(sys => sys.page === p);
        if (0 <= s && s < pSys.length) {
          targetSystem = pSys[s];
          targetSystemIndex = (currentParsedSheet.systems || []).indexOf(targetSystem);
        }
      }
    }
    if (!targetSystem && (currentParsedSheet.systems || []).length > 0) {
      targetSystem = currentParsedSheet.systems[0];
      targetSystemIndex = 0;
    }

    const header = document.createElement("div");
    header.className = "qa-review-header";
    header.innerHTML = `
      <div class="qa-review-title">
        <span class="qa-badge qa-badge-structural">结构/小节线</span>
        <span>${issue.message}</span>
      </div>
      <span style="font-size: 0.8rem; color: #9a3412;">建议检查该整行乐谱的小节线划分与小节总数</span>
    `;
    card.appendChild(header);

    if (targetSystem) {
      const cropContainer = document.createElement("div");
      cropContainer.className = "qa-crop-container";

      const canvasWrapper = document.createElement("div");
      canvasWrapper.className = "qa-crop-canvas-wrapper scrollable-row";
      canvasWrapper.style.width = "100%";

      const canvas = document.createElement("canvas");
      canvas.className = "qa-crop-canvas";
      canvasWrapper.appendChild(canvas);
      cropContainer.appendChild(canvasWrapper);

      const rowMeasureCount = targetSystem && targetSystem.measures ? targetSystem.measures.length : 4;
      const hint = document.createElement("div");
      hint.className = "qa-crop-hint";
      hint.innerHTML = `<span>🔍 点击整行截图可在高清大图中自由缩放平移（当前行检测到 ${rowMeasureCount} 个小节，蓝色竖线为识别的小节线）</span><span class="qa-mobile-scroll-hint"> ← 左右滑动可完整查看 →</span>`;
      cropContainer.appendChild(hint);

      card.appendChild(cropContainer);

      const pageImg = new Image();
      pageImg.src = `/api/pages/${currentSheetId}/${targetSystem.page}`;
      pageImg.onload = () => {
        const natW = pageImg.naturalWidth;
        const natH = pageImg.naturalHeight;
        const sBbox = targetSystem.bbox;
        const padY = Math.round(natH * 0.012);
        const sx = 0;
        const sy = Math.max(0, Math.round(sBbox[1] * natH) - padY);
        const sw = natW;
        const sh = Math.min(natH - sy, Math.round((sBbox[3] - sBbox[1]) * natH) + padY * 2);

        canvas.width = Math.max(sw, 1000);
        canvas.height = Math.round(sh * (canvas.width / sw));

        const ctx = canvas.getContext("2d");
        ctx.drawImage(pageImg, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

        const measures = targetSystem.measures || [];
        ctx.strokeStyle = "#2563eb";
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 4]);

        measures.forEach(m => {
          const lineX = Math.round((m.bbox[0] * natW - sx) * (canvas.width / sw));
          ctx.beginPath();
          ctx.moveTo(lineX, 0);
          ctx.lineTo(lineX, canvas.height);
          ctx.stroke();

          ctx.save();
          ctx.setLineDash([]);
          ctx.fillStyle = "#1d4ed8";
          ctx.fillRect(lineX + 2, 4, 74, 22);
          ctx.fillStyle = "#ffffff";
          ctx.font = "bold 13px sans-serif";
          ctx.fillText(`第 ${m.index + 1} 节`, lineX + 6, 20);
          ctx.restore();
        });

        if (measures.length > 0) {
          const lastM = measures[measures.length - 1];
          const endX = Math.round((lastM.bbox[2] * natW - sx) * (canvas.width / sw));
          ctx.beginPath();
          ctx.moveTo(endX, 0);
          ctx.lineTo(endX, canvas.height);
          ctx.stroke();
        }

        canvasWrapper.onclick = () => {
          openLightbox(`第 ${targetSystem.page + 1} 页第 ${targetSystemIndex + 1} 行原谱高清截图`, (lbCanvas) => {
            lbCanvas.width = canvas.width;
            lbCanvas.height = canvas.height;
            const lbCtx = lbCanvas.getContext("2d");
            lbCtx.drawImage(canvas, 0, 0);
          });
        };
      };
    }

    const actionsBar = document.createElement("div");
    actionsBar.className = "qa-card-actions";

    const confirmBtn = document.createElement("button");
    confirmBtn.type = "button";
    confirmBtn.className = "btn btn-success btn-sm";
    confirmBtn.textContent = "✓ 划分正确 (确认)";
    confirmBtn.onclick = async () => {
      await confirmIssue({
        issue_code: issue.code,
        measure_index: issue.measure_index,
        action: "confirm",
      });
    };
    actionsBar.appendChild(confirmBtn);

    const corrGroup = document.createElement("div");
    corrGroup.style.display = "inline-flex";
    corrGroup.style.alignItems = "center";
    corrGroup.style.gap = "6px";

    const numLabel = document.createElement("span");
    numLabel.style.fontSize = "0.85rem";
    numLabel.style.color = "#7c2d12";
    numLabel.style.fontWeight = "600";
    numLabel.textContent = "若数量有误：";
    corrGroup.appendChild(numLabel);

    const numInput = document.createElement("input");
    numInput.type = "number";
    numInput.min = "1";
    numInput.max = "16";
    numInput.className = "form-control";
    numInput.style.width = "64px";
    numInput.style.padding = "4px 8px";
    const rowCount = (targetSystem && targetSystem.measures) ? targetSystem.measures.length : 4;
    numInput.value = rowCount;
    corrGroup.appendChild(numInput);

    const corrBtn = document.createElement("button");
    corrBtn.type = "button";
    corrBtn.className = "btn btn-secondary btn-sm";
    corrBtn.textContent = "小节数应为该值";
    corrBtn.onclick = async () => {
      const val = parseInt(numInput.value, 10);
      if (!val || val <= 0) {
        alert("请输入有效的小节数 (大于0)");
        return;
      }
      const noteText = `第${targetSystem ? targetSystem.page + 1 : 1}页第${targetSystemIndex + 1}行小节数更正为 ${val}`;
      await confirmIssue({
        issue_code: issue.code,
        measure_index: issue.measure_index,
        action: "correct",
        measure_count: val,
        note: noteText,
      });
    };
    corrGroup.appendChild(corrBtn);
    actionsBar.appendChild(corrGroup);

    card.appendChild(actionsBar);
    return card;
  }

  function buildChordCard(issue, issueIdx) {
    const card = document.createElement("div");
    card.className = "qa-review-card chord";

    const mIdx = issue.measure_index;
    let targetMeasure = null;
    let targetSystem = null;
    let targetChord = null;

    (currentParsedSheet.systems || []).forEach(sys => {
      (sys.measures || []).forEach(m => {
        if (m.index === mIdx) {
          targetMeasure = m;
          targetSystem = sys;
          if (m.chords && m.chords.length > 0) {
            if (issue.detail && issue.detail.chord_index !== undefined && m.chords[issue.detail.chord_index]) {
              targetChord = m.chords[issue.detail.chord_index];
            } else if (issue.detail && issue.detail.beat !== undefined) {
              targetChord = m.chords.find(c => Math.abs(c.beat - issue.detail.beat) < 0.01) || m.chords[0];
            } else if (issue.detail && issue.detail.raw) {
              targetChord = m.chords.find(c => c.raw === issue.detail.raw) || m.chords[0];
            } else {
              targetChord = m.chords[0];
            }
          }
        }
      });
    });

    const header = document.createElement("div");
    header.className = "qa-review-header";
    header.innerHTML = `
      <div class="qa-review-title">
        <span class="qa-badge qa-badge-chord">和弦核对</span>
        <span>第 ${mIdx !== null && mIdx !== undefined ? mIdx + 1 : '—'} 小节：${issue.message}</span>
      </div>
    `;
    card.appendChild(header);

    if (targetSystem) {
      const cropContainer = document.createElement("div");
      cropContainer.className = "qa-crop-container";

      const canvasWrapper = document.createElement("div");
      canvasWrapper.className = "qa-crop-canvas-wrapper";

      const canvas = document.createElement("canvas");
      canvas.className = "qa-crop-canvas";
      canvasWrapper.appendChild(canvas);
      cropContainer.appendChild(canvasWrapper);

      const hint = document.createElement("div");
      hint.className = "qa-crop-hint";
      hint.innerHTML = `<span>🔍 点击原谱截图可在高清大图中查看和弦与旋律（红框标出和弦框，含左右邻近上下文）</span>`;
      cropContainer.appendChild(hint);

      card.appendChild(cropContainer);

      const pageImg = new Image();
      pageImg.src = `/api/pages/${currentSheetId}/${targetSystem.page}`;
      pageImg.onload = () => {
        const natW = pageImg.naturalWidth;
        const natH = pageImg.naturalHeight;
        const mBbox = (targetMeasure && targetMeasure.bbox) ? targetMeasure.bbox : [0.1, targetSystem.bbox[1], 0.3, targetSystem.bbox[3]];
        const mw = mBbox[2] - mBbox[0];
        const x0 = Math.max(0, mBbox[0] - mw * 0.3);
        const x1 = Math.min(1, mBbox[2] + mw * 0.3);
        const y0 = Math.max(0, targetSystem.bbox[1] - 0.005);
        const y1 = Math.min(1, targetSystem.bbox[3] + 0.005);

        const srcX = Math.round(x0 * natW);
        const srcY = Math.round(y0 * natH);
        const srcW = Math.round((x1 - x0) * natW);
        const srcH = Math.round((y1 - y0) * natH);

        const dispW = Math.max(srcW * 3, 480);
        const dispH = Math.round(srcH * (dispW / srcW));
        const scale = dispW / srcW;

        canvas.width = Math.round(dispW);
        canvas.height = Math.round(dispH);
        canvas.style.width = `${Math.round(dispW)}px`;
        canvas.style.maxWidth = "100%";
        canvas.style.height = "auto";

        const ctx = canvas.getContext("2d");
        if (scale <= 4) ctx.imageSmoothingEnabled = false;
        ctx.drawImage(pageImg, srcX, srcY, srcW, srcH, 0, 0, canvas.width, canvas.height);

        const chordBbox = (targetChord && targetChord.bbox)
          ? targetChord.bbox
          : (issue.detail && issue.detail.bbox ? issue.detail.bbox : null);

        if (chordBbox) {
          const cbX = Math.round((chordBbox[0] * natW - srcX) * (canvas.width / srcW));
          const cbY = Math.round((chordBbox[1] * natH - srcY) * (canvas.height / srcH));
          const cbW = Math.round((chordBbox[2] - chordBbox[0]) * natW * (canvas.width / srcW));
          const cbH = Math.round((chordBbox[3] - chordBbox[1]) * natH * (canvas.height / srcH));
          ctx.strokeStyle = "#dc2626";
          ctx.lineWidth = 3.5;
          ctx.strokeRect(cbX, cbY, cbW, cbH);

          ctx.fillStyle = "#dc2626";
          ctx.fillRect(cbX, Math.max(0, cbY - 18), 54, 18);
          ctx.fillStyle = "#ffffff";
          ctx.font = "bold 11px sans-serif";
          ctx.fillText("和弦框", cbX + 4, Math.max(13, cbY - 4));
        } else {
          const mX = Math.round((mBbox[0] * natW - srcX) * (canvas.width / srcW));
          const mW = Math.round((mBbox[2] - mBbox[0]) * natW * (canvas.width / srcW));
          ctx.strokeStyle = "#ea580c";
          ctx.lineWidth = 2.5;
          ctx.setLineDash([4, 4]);
          ctx.strokeRect(mX, 2, mW, canvas.height - 4);
          ctx.setLineDash([]);
        }

        canvasWrapper.onclick = () => {
          openLightbox(`第 ${mIdx + 1} 小节和弦原谱高清截图`, (lbCanvas) => {
            lbCanvas.width = canvas.width;
            lbCanvas.height = canvas.height;
            const lbCtx = lbCanvas.getContext("2d");
            lbCtx.drawImage(canvas, 0, 0);
          });
        };
      };
    }

    const actionsBar = document.createElement("div");
    actionsBar.className = "qa-card-actions";

    const alts = (targetChord && targetChord.alternatives && targetChord.alternatives.length > 0)
      ? targetChord.alternatives
      : (issue.detail && issue.detail.alternatives ? issue.detail.alternatives : []);

    if (alts.length > 0) {
      const altLabel = document.createElement("span");
      altLabel.style.fontSize = "0.85rem";
      altLabel.style.fontWeight = "600";
      altLabel.style.color = "#92400e";
      altLabel.textContent = "候选和弦：";
      actionsBar.appendChild(altLabel);

      alts.forEach(alt => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "qa-alt-btn";
        btn.textContent = `选用 "${alt}"`;
        btn.onclick = async () => {
          await confirmIssue({
            issue_code: issue.code,
            measure_index: mIdx,
            chord: alt,
          });
        };
        actionsBar.appendChild(btn);
      });
    }

    const freeGroup = document.createElement("div");
    freeGroup.style.display = "inline-flex";
    freeGroup.style.alignItems = "center";
    freeGroup.style.gap = "6px";

    const freeInput = document.createElement("input");
    freeInput.type = "text";
    freeInput.className = "form-control";
    freeInput.style.width = "120px";
    freeInput.style.padding = "4px 8px";
    freeInput.placeholder = "自由输入和弦";

    const freeBtn = document.createElement("button");
    freeBtn.type = "button";
    freeBtn.className = "btn btn-primary btn-sm";
    freeBtn.textContent = "修改为该和弦";
    freeBtn.onclick = async () => {
      const val = freeInput.value.trim();
      if (!val) {
        alert("请输入和弦名称");
        return;
      }
      await confirmIssue({
        issue_code: issue.code,
        measure_index: mIdx,
        chord: val,
      });
    };
    freeGroup.appendChild(freeInput);
    freeGroup.appendChild(freeBtn);
    actionsBar.appendChild(freeGroup);

    const curChord = (targetChord && targetChord.raw) ? targetChord.raw : (issue.detail && issue.detail.raw ? issue.detail.raw : "当前");
    const confirmBtn = document.createElement("button");
    confirmBtn.type = "button";
    confirmBtn.className = "btn btn-secondary btn-sm qa-confirm-chord-btn";
    confirmBtn.textContent = `确认 "${curChord}" 正确`;
    confirmBtn.onclick = async () => {
      await confirmIssue({
        issue_code: issue.code,
        measure_index: mIdx,
        action: "confirm",
        chord: targetChord ? targetChord.raw : undefined,
      });
    };
    actionsBar.appendChild(confirmBtn);

    card.appendChild(actionsBar);
    return card;
  }

  function buildKeyChangeCard(issue, issueIdx) {
    const card = document.createElement("div");
    card.className = "qa-review-card key-change";

    const header = document.createElement("div");
    header.className = "qa-review-header";
    header.innerHTML = `
      <div class="qa-review-title">
        <span class="qa-badge qa-badge-key">转调设置</span>
        <span>${issue.message}</span>
      </div>
      <span style="font-size: 0.8rem; color: #6b21a8;">谱头已标记转调，请选择转调起始小节与半音差</span>
    `;
    card.appendChild(header);

    const controlsRow = document.createElement("div");
    controlsRow.className = "qa-card-actions";
    controlsRow.style.borderTop = "none";
    controlsRow.style.paddingTop = "0";

    const mLabel = document.createElement("span");
    mLabel.style.fontSize = "0.85rem";
    mLabel.style.fontWeight = "600";
    mLabel.textContent = "转调起始小节：";
    controlsRow.appendChild(mLabel);

    const mSelect = document.createElement("select");
    mSelect.className = "form-control";
    mSelect.style.width = "140px";
    const allMeasures = [];
    (currentParsedSheet.systems || []).forEach(s => (s.measures || []).forEach(m => allMeasures.push(m)));
    allMeasures.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.index;
      opt.textContent = `第 ${m.index + 1} 小节`;
      mSelect.appendChild(opt);
    });
    if (issue.measure_index !== null && issue.measure_index !== undefined) {
      mSelect.value = issue.measure_index;
    } else if (allMeasures.length > 10) {
      mSelect.value = Math.floor(allMeasures.length / 2);
    }
    controlsRow.appendChild(mSelect);

    const sLabel = document.createElement("span");
    sLabel.style.fontSize = "0.85rem";
    sLabel.style.fontWeight = "600";
    sLabel.style.marginLeft = "8px";
    sLabel.textContent = "半音差：";
    controlsRow.appendChild(sLabel);

    const sSelect = document.createElement("select");
    sSelect.className = "form-control";
    sSelect.style.width = "170px";
    const semiOptions = [
      { val: 2, label: "+2 (升全音，如 D→E, F#→Ab)" },
      { val: 1, label: "+1 (升半音，如 C→Db)" },
      { val: 3, label: "+3 (小三度，如 A→C)" },
      { val: 4, label: "+4 (大三度，如 C→E)" },
      { val: 5, label: "+5 (纯四度，如 C→F)" },
      { val: -1, label: "-1 (降半音)" },
      { val: -2, label: "-2 (降全音)" },
    ];
    semiOptions.forEach(optData => {
      const opt = document.createElement("option");
      opt.value = optData.val;
      opt.textContent = optData.label;
      if (optData.val === 2) opt.selected = true;
      sSelect.appendChild(opt);
    });
    controlsRow.appendChild(sSelect);

    const applyBtn = document.createElement("button");
    applyBtn.type = "button";
    applyBtn.className = "btn btn-primary btn-sm";
    applyBtn.textContent = "确认设置转调";
    applyBtn.onclick = async () => {
      const atM = parseInt(mSelect.value, 10);
      const semi = parseInt(sSelect.value, 10);
      await confirmIssue({
        issue_code: issue.code,
        at_measure: atM,
        semitones: semi,
        action: "confirm",
        note: `转调 (${semi > 0 ? '+' : ''}${semi}半音)`,
      });
    };
    controlsRow.appendChild(applyBtn);

    const ignoreBtn = document.createElement("button");
    ignoreBtn.type = "button";
    ignoreBtn.className = "btn btn-secondary btn-sm";
    ignoreBtn.textContent = "无需转调 (忽略)";
    ignoreBtn.onclick = async () => {
      await confirmIssue({
        issue_code: issue.code,
        action: "confirm",
      });
    };
    controlsRow.appendChild(ignoreBtn);

    card.appendChild(controlsRow);

    const cropContainer = document.createElement("div");
    cropContainer.className = "qa-crop-container";

    const canvasWrapper = document.createElement("div");
    canvasWrapper.className = "qa-crop-canvas-wrapper";
    canvasWrapper.style.width = "100%";

    const canvas = document.createElement("canvas");
    canvas.className = "qa-crop-canvas";
    canvasWrapper.appendChild(canvas);
    cropContainer.appendChild(canvasWrapper);

    const hint = document.createElement("div");
    hint.className = "qa-crop-hint";
    hint.innerHTML = `<span>🔍 点击原谱行截图可放大查看转调标记位置（如"轉成2調"、"Mod."）</span>`;
    cropContainer.appendChild(hint);

    card.appendChild(cropContainer);

    function updateRowCrop() {
      const selMIdx = parseInt(mSelect.value, 10);
      let targetSys = null;
      let sIdx = 0;
      (currentParsedSheet.systems || []).forEach((sys, idx) => {
        if (sys.measures && sys.measures.some(m => m.index === selMIdx)) {
          targetSys = sys;
          sIdx = idx;
        }
      });
      if (!targetSys && (currentParsedSheet.systems || []).length > 0) {
        targetSys = currentParsedSheet.systems[0];
      }
      if (!targetSys) return;

      const pageImg = new Image();
      pageImg.src = `/api/pages/${currentSheetId}/${targetSys.page}`;
      pageImg.onload = () => {
        const natW = pageImg.naturalWidth;
        const natH = pageImg.naturalHeight;
        const sBbox = targetSys.bbox;
        const padY = Math.round(natH * 0.012);
        const sx = 0;
        const sy = Math.max(0, Math.round(sBbox[1] * natH) - padY);
        const sw = natW;
        const sh = Math.min(natH - sy, Math.round((sBbox[3] - sBbox[1]) * natH) + padY * 2);

        canvas.width = Math.max(sw, 960);
        canvas.height = Math.round(sh * (canvas.width / sw));

        const ctx = canvas.getContext("2d");
        ctx.drawImage(pageImg, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

        const selM = (targetSys.measures || []).find(m => m.index === selMIdx);
        if (selM) {
          const mx = Math.round((selM.bbox[0] * natW - sx) * (canvas.width / sw));
          const mw = Math.round((selM.bbox[2] - selM.bbox[0]) * natW * (canvas.width / sw));
          ctx.strokeStyle = "#9333ea";
          ctx.lineWidth = 3;
          ctx.strokeRect(mx, 2, mw, canvas.height - 4);
          ctx.fillStyle = "rgba(147, 51, 234, 0.15)";
          ctx.fillRect(mx, 2, mw, canvas.height - 4);
        }

        canvasWrapper.onclick = () => {
          openLightbox(`第 ${targetSys.page + 1} 页第 ${sIdx + 1} 行原谱高清图`, (lbCanvas) => {
            lbCanvas.width = canvas.width;
            lbCanvas.height = canvas.height;
            const lbCtx = lbCanvas.getContext("2d");
            lbCtx.drawImage(canvas, 0, 0);
          });
        };
      };
    }

    mSelect.onchange = updateRowCrop;
    updateRowCrop();

    return card;
  }

  function buildMelodyBeatCard(issue, issueIdx) {
    const card = document.createElement("div");
    card.className = "qa-review-card melody-beat";

    const mIdx = issue.measure_index;
    let targetMeasure = null;
    let targetSystem = null;

    (currentParsedSheet.systems || []).forEach(sys => {
      (sys.measures || []).forEach(m => {
        if (m.index === mIdx) {
          targetMeasure = m;
          targetSystem = sys;
        }
      });
    });

    const header = document.createElement("div");
    header.className = "qa-review-header";
    header.innerHTML = `
      <div class="qa-review-title">
        <span class="qa-badge qa-badge-melody">旋律/拍数</span>
        <span>第 ${mIdx !== null && mIdx !== undefined ? mIdx + 1 : '—'} 小节：${issue.message}</span>
      </div>
    `;
    card.appendChild(header);

    if (targetSystem) {
      const cropContainer = document.createElement("div");
      cropContainer.className = "qa-crop-container";

      const canvasWrapper = document.createElement("div");
      canvasWrapper.className = "qa-crop-canvas-wrapper";
      canvasWrapper.style.width = "100%";

      const canvas = document.createElement("canvas");
      canvas.className = "qa-crop-canvas";
      canvasWrapper.appendChild(canvas);
      cropContainer.appendChild(canvasWrapper);

      const hint = document.createElement("div");
      hint.className = "qa-crop-hint";
      hint.innerHTML = `<span>🔍 点击原谱行截图可在高清大图中查看旋律音符与小节划分</span>`;
      cropContainer.appendChild(hint);

      card.appendChild(cropContainer);

      const pageImg = new Image();
      pageImg.src = `/api/pages/${currentSheetId}/${targetSystem.page}`;
      pageImg.onload = () => {
        const natW = pageImg.naturalWidth;
        const natH = pageImg.naturalHeight;
        const sBbox = targetSystem.bbox;
        const padY = Math.round(natH * 0.01);
        const sx = 0;
        const sy = Math.max(0, Math.round(sBbox[1] * natH) - padY);
        const sw = natW;
        const sh = Math.min(natH - sy, Math.round((sBbox[3] - sBbox[1]) * natH) + padY * 2);

        canvas.width = Math.max(sw, 960);
        canvas.height = Math.round(sh * (canvas.width / sw));

        const ctx = canvas.getContext("2d");
        ctx.drawImage(pageImg, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

        if (targetMeasure && targetMeasure.bbox) {
          const mx = Math.round((targetMeasure.bbox[0] * natW - sx) * (canvas.width / sw));
          const mw = Math.round((targetMeasure.bbox[2] - targetMeasure.bbox[0]) * natW * (canvas.width / sw));
          ctx.strokeStyle = "#2563eb";
          ctx.lineWidth = 3;
          ctx.strokeRect(mx, 2, mw, canvas.height - 4);
          ctx.fillStyle = "rgba(37, 99, 235, 0.1)";
          ctx.fillRect(mx, 2, mw, canvas.height - 4);
        }

        canvasWrapper.onclick = () => {
          openLightbox(`第 ${mIdx + 1} 小节旋律原谱高清大图`, (lbCanvas) => {
            lbCanvas.width = canvas.width;
            lbCanvas.height = canvas.height;
            const lbCtx = lbCanvas.getContext("2d");
            lbCtx.drawImage(canvas, 0, 0);
          });
        };
      };
    }

    const actionsBar = document.createElement("div");
    actionsBar.className = "qa-card-actions";

    const chords = targetMeasure ? (targetMeasure.chords || []) : [];
    if (chords.length > 0) {
      const c = chords[0];
      const bLabel = document.createElement("span");
      bLabel.style.fontSize = "0.85rem";
      bLabel.style.fontWeight = "600";
      bLabel.textContent = `和弦 "${c.raw}" 起始拍：`;
      actionsBar.appendChild(bLabel);

      const bSelect = document.createElement("select");
      bSelect.className = "form-control";
      bSelect.style.width = "100px";
      [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0].forEach(bt => {
        const opt = document.createElement("option");
        opt.value = bt;
        opt.textContent = `第 ${bt} 拍`;
        if (Math.abs(c.beat - bt) < 0.1) opt.selected = true;
        bSelect.appendChild(opt);
      });
      actionsBar.appendChild(bSelect);

      const saveBeatBtn = document.createElement("button");
      saveBeatBtn.type = "button";
      saveBeatBtn.className = "btn btn-primary btn-sm";
      saveBeatBtn.textContent = "更正起始拍";
      saveBeatBtn.onclick = async () => {
        await confirmIssue({
          issue_code: issue.code,
          measure_index: mIdx,
          beat: parseFloat(bSelect.value),
          action: "confirm",
        });
      };
      actionsBar.appendChild(saveBeatBtn);
    }

    const confirmBtn = document.createElement("button");
    confirmBtn.type = "button";
    confirmBtn.className = "btn btn-secondary btn-sm";
    confirmBtn.textContent = "拍数确认无误";
    confirmBtn.onclick = async () => {
      await confirmIssue({
        issue_code: issue.code,
        measure_index: mIdx,
        action: "confirm",
      });
    };
    actionsBar.appendChild(confirmBtn);

    card.appendChild(actionsBar);
    return card;
  }

  function buildMissingChordCard(issue, issueIdx) {
    const card = document.createElement("div");
    card.className = "qa-review-card chord";

    const mIdx = issue.measure_index;
    let targetMeasure = null;
    let targetSystem = null;

    (currentParsedSheet.systems || []).forEach(sys => {
      (sys.measures || []).forEach(m => {
        if (m.index === mIdx) {
          targetMeasure = m;
          targetSystem = sys;
        }
      });
    });

    const header = document.createElement("div");
    header.className = "qa-review-header";
    header.innerHTML = `
      <div class="qa-review-title">
        <span class="qa-badge qa-badge-chord">疑似遗漏和弦</span>
        <span>第 ${mIdx !== null && mIdx !== undefined ? mIdx + 1 : '—'} 小节：${issue.message}</span>
      </div>
    `;
    card.appendChild(header);

    if (targetSystem) {
      const cropContainer = document.createElement("div");
      cropContainer.className = "qa-crop-container";

      const canvasWrapper = document.createElement("div");
      canvasWrapper.className = "qa-crop-canvas-wrapper";

      const canvas = document.createElement("canvas");
      canvas.className = "qa-crop-canvas";
      canvasWrapper.appendChild(canvas);
      cropContainer.appendChild(canvasWrapper);

      const hint = document.createElement("div");
      hint.className = "qa-crop-hint";
      hint.innerHTML = `<span>🔍 点击截图可在高清大图中查看原谱小节印迹</span>`;
      cropContainer.appendChild(hint);

      card.appendChild(cropContainer);

      const pageImg = new Image();
      pageImg.src = `/api/pages/${currentSheetId}/${targetSystem.page}`;
      pageImg.onload = () => {
        const natW = pageImg.naturalWidth;
        const natH = pageImg.naturalHeight;
        const mBbox = (targetMeasure && targetMeasure.bbox) ? targetMeasure.bbox : [0.1, targetSystem.bbox[1], 0.3, targetSystem.bbox[3]];
        const mw = mBbox[2] - mBbox[0];
        const x0 = Math.max(0, mBbox[0] - mw * 0.3);
        const x1 = Math.min(1, mBbox[2] + mw * 0.3);
        const y0 = Math.max(0, targetSystem.bbox[1] - 0.005);
        const y1 = Math.min(1, targetSystem.bbox[3] + 0.005);

        const srcX = Math.round(x0 * natW);
        const srcY = Math.round(y0 * natH);
        const srcW = Math.round((x1 - x0) * natW);
        const srcH = Math.round((y1 - y0) * natH);

        const dispW = Math.max(srcW * 3, 480);
        const dispH = Math.round(srcH * (dispW / srcW));
        const scale = dispW / srcW;

        canvas.width = Math.round(dispW);
        canvas.height = Math.round(dispH);
        canvas.style.width = `${Math.round(dispW)}px`;
        canvas.style.maxWidth = "100%";
        canvas.style.height = "auto";

        const ctx = canvas.getContext("2d");
        if (scale <= 4) ctx.imageSmoothingEnabled = false;
        ctx.drawImage(pageImg, srcX, srcY, srcW, srcH, 0, 0, canvas.width, canvas.height);

        const mx = Math.round((mBbox[0] * natW - srcX) * (canvas.width / srcW));
        const mwPix = Math.round((mBbox[2] - mBbox[0]) * natW * (canvas.width / srcW));
        ctx.strokeStyle = "#d97706";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(mx, 2, mwPix, canvas.height - 4);
        ctx.setLineDash([]);

        canvasWrapper.onclick = () => {
          openLightbox(`第 ${mIdx + 1} 小节原谱高清截图`, (lbCanvas) => {
            lbCanvas.width = canvas.width;
            lbCanvas.height = canvas.height;
            const lbCtx = lbCanvas.getContext("2d");
            lbCtx.drawImage(canvas, 0, 0);
          });
        };
      };
    }

    const actionsBar = document.createElement("div");
    actionsBar.className = "qa-card-actions";

    const cand = issue.detail && issue.detail.candidate ? issue.detail.candidate : null;
    const candBeat = issue.detail && issue.detail.beat ? issue.detail.beat : 3.0;

    if (cand) {
      const addCandBtn = document.createElement("button");
      addCandBtn.type = "button";
      addCandBtn.className = "btn btn-primary btn-sm";
      addCandBtn.textContent = `添加和弦 "${cand}" (第 ${candBeat} 拍)`;
      addCandBtn.onclick = async () => {
        await confirmIssue({
          issue_code: issue.code,
          measure_index: mIdx,
          chord: cand,
          beat: candBeat,
          action: "confirm",
        });
      };
      actionsBar.appendChild(addCandBtn);
    }

    const freeGroup = document.createElement("div");
    freeGroup.style.display = "inline-flex";
    freeGroup.style.alignItems = "center";
    freeGroup.style.gap = "6px";

    const fInput = document.createElement("input");
    fInput.type = "text";
    fInput.className = "form-control";
    fInput.style.width = "100px";
    fInput.placeholder = "补充和弦";

    const fBeat = document.createElement("input");
    fBeat.type = "number";
    fBeat.step = "0.5";
    fBeat.className = "form-control";
    fBeat.style.width = "60px";
    fBeat.value = "3.0";

    const fBtn = document.createElement("button");
    fBtn.type = "button";
    fBtn.className = "btn btn-secondary btn-sm";
    fBtn.textContent = "添加此和弦";
    fBtn.onclick = async () => {
      const chordVal = fInput.value.trim();
      if (!chordVal) {
        alert("请输入和弦");
        return;
      }
      const beatVal = parseFloat(fBeat.value) || 1.0;
      await confirmIssue({
        issue_code: issue.code,
        measure_index: mIdx,
        chord: chordVal,
        beat: beatVal,
        action: "confirm",
      });
    };
    freeGroup.appendChild(fInput);
    freeGroup.appendChild(fBeat);
    freeGroup.appendChild(fBtn);
    actionsBar.appendChild(freeGroup);

    const noMissBtn = document.createElement("button");
    noMissBtn.type = "button";
    noMissBtn.className = "btn btn-secondary btn-sm";
    noMissBtn.textContent = "确认无遗漏 (忽略)";
    noMissBtn.onclick = async () => {
      await confirmIssue({
        issue_code: issue.code,
        measure_index: mIdx,
        action: "confirm",
      });
    };
    actionsBar.appendChild(noMissBtn);

    card.appendChild(actionsBar);
    return card;
  }

  // --- Step 3: Review & Edit ---
  function renderStep3() {
    if (!currentParsedSheet) return;

    const hdr = currentParsedSheet.header || {};
    hdrTitle.value = hdr.title || "";
    hdrStyle.value = hdr.style || "";
    hdrTempo.value = hdr.tempo_bpm || "";
    hdrTimesig.value = hdr.time_signature || "4/4";

    // QA Review Summary & Cards
    const issues = currentParsedSheet.issues || [];
    const needsReview = issues.filter(i => i.severity === "needs_review");
    const autoFixed = issues.filter(i => i.severity === "auto_fixed");
    const N = autoFixed.length;
    const M = needsReview.length;

    if (qaSummaryBar) {
      qaSummaryBar.classList.remove("hidden", "has-issues", "all-clean");
      if (M > 0) {
        qaSummaryBar.classList.add("has-issues");
      } else {
        qaSummaryBar.classList.add("all-clean");
      }
      qaSummaryText.textContent = `自动校验：修正 ${N} 处，需要您确认 ${M} 处`;
    }

    const genLabel = M > 0 ? `生成（仍有 ${M} 处未确认） 🎶` : "生成伴奏谱 🎶";
    btnTriggerRender.textContent = genLabel;
    btnTriggerRenderBottom.textContent = genLabel;

    if (measuresDetails) {
      measuresDetails.open = (M > 0);
    }

    // Auto-fixed section
    if (qaAutofixedDetails) {
      if (N > 0) {
        qaAutofixedDetails.classList.remove("hidden");
        qaAutofixedSummary.textContent = `已自动修正 (${N} 处)`;
        qaAutofixedList.innerHTML = "";
        autoFixed.forEach(issue => {
          const item = document.createElement("div");
          item.className = "qa-autofixed-item";
          const mText = issue.measure_index !== null && issue.measure_index !== undefined
            ? `第 ${issue.measure_index + 1} 小节` : "全局";
          item.innerHTML = `<span><strong>${mText}</strong>: ${issue.message}</span>`;
          qaAutofixedList.appendChild(item);
        });
      } else {
        qaAutofixedDetails.classList.add("hidden");
      }
    }

    // Layout Gate Check
    hideRenderError();
    const layoutConf = currentParsedSheet.layout_confidence !== undefined ? currentParsedSheet.layout_confidence : 1.0;
    const allMeasures = [];
    (currentParsedSheet.systems || []).forEach(s => (s.measures || []).forEach(m => allMeasures.push(m)));
    const totalMeasures = allMeasures.length;
    const reviewMeasures = new Set(needsReview.map(i => i.measure_index).filter(idx => idx !== null && idx !== undefined));
    const reviewRatio = totalMeasures > 0 ? (reviewMeasures.size / totalMeasures) : 0;
    const isLowConf = layoutConf < 0.6;
    const isHighReview = reviewRatio > 0.25;

    const unconfirmedStructural = needsReview.filter(i => isStructuralIssue(i));
    const structuralConfirmed = (currentParsedSheet.warnings || []).some(w => w.includes("已确认版面结构") || w.includes("structural_confirmed"));

    const isGated = (isLowConf && (unconfirmedStructural.length > 0 || !structuralConfirmed)) || (isHighReview && unconfirmedStructural.length > 0);

    if (qaLayoutGateBanner) {
      if (isGated) {
        qaLayoutGateBanner.classList.remove("hidden");
        const reasons = [];
        if (isLowConf) {
          reasons.push(`版面模型置信度偏低（${(layoutConf * 100).toFixed(0)}% < 60%）`);
        }
        if (isHighReview) {
          reasons.push(`待核对小节比例偏高（${(reviewRatio * 100).toFixed(0)}% > 25%，共 ${reviewMeasures.size}/${totalMeasures} 小节）`);
        }
        if (unconfirmedStructural.length > 0) {
          reasons.push(`存在 ${unconfirmedStructural.length} 处未核对的小节线/结构划分`);
        }
        qaLayoutGateDesc.textContent = `原因：${reasons.join("；")}。请在下方核对卡片中确认或更正小节线与行划分后即可生成伴奏。`;

        // Problem rows
        qaLayoutGateDetails.innerHTML = "";
        const problemSystems = new Set();
        needsReview.forEach(i => {
          (currentParsedSheet.systems || []).forEach((sys, sIdx) => {
            if (i.measure_index !== null && i.measure_index !== undefined && sys.measures && sys.measures.some(m => m.index === i.measure_index)) {
              problemSystems.add(`第 ${sys.page + 1} 页第 ${sIdx + 1} 行`);
            }
          });
          const mMatch = /第\s*(\d+)\s*页第\s*(\d+)\s*行/.exec(i.message);
          if (mMatch) {
            problemSystems.add(`第 ${mMatch[1]} 页第 ${mMatch[2]} 行`);
          }
        });

        if (problemSystems.size > 0) {
          const rowContainer = document.createElement("div");
          rowContainer.style.marginTop = "6px";
          rowContainer.innerHTML = `<strong>涉及问题行：</strong>` + Array.from(problemSystems).map(r => `<span class="qa-problem-row-tag">${r}</span>`).join("");
          qaLayoutGateDetails.appendChild(rowContainer);
        }

        btnTriggerRender.disabled = true;
        btnTriggerRenderBottom.disabled = true;
        btnTriggerRender.title = "请先确认下方版面结构问题";
        btnTriggerRenderBottom.title = "请先确认下方版面结构问题";
      } else {
        qaLayoutGateBanner.classList.add("hidden");
        btnTriggerRender.disabled = false;
        btnTriggerRenderBottom.disabled = false;
        btnTriggerRender.removeAttribute("title");
        btnTriggerRenderBottom.removeAttribute("title");
      }
    }

    // Needs review section
    if (qaNeedsReviewSection) {
      if (M > 0) {
        qaNeedsReviewSection.classList.remove("hidden");
        qaNeedsReviewList.innerHTML = "";

        needsReview.forEach((issue, issueIdx) => {
          const type = getIssueType(issue);
          let card;
          if (type === "structural") {
            card = buildStructuralCard(issue, issueIdx);
          } else if (type === "key_change") {
            card = buildKeyChangeCard(issue, issueIdx);
          } else if (type === "melody_beat") {
            card = buildMelodyBeatCard(issue, issueIdx);
          } else if (type === "missing_chord") {
            card = buildMissingChordCard(issue, issueIdx);
          } else {
            card = buildChordCard(issue, issueIdx);
          }
          qaNeedsReviewList.appendChild(card);
        });
      } else {
        qaNeedsReviewSection.classList.add("hidden");
      }
    }

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
        const isNeedsReview = (currentParsedSheet.issues || []).some(
          i => i.severity === "needs_review" && i.measure_index === m.index
        );
        const mBox = document.createElement("div");
        mBox.className = `measure-box ${isNeedsReview ? 'needs-review' : ''}`;

        const mNum = document.createElement("div");
        mNum.className = "measure-num";
        const badge = isNeedsReview ? ' <span style="color:#d97706;font-size:0.75rem;">[待确认]</span>' : '';
        mNum.innerHTML = `<span>小节 #${m.index + 1}${badge}</span> <span>${m.beats || 4} 拍</span>`;
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
            c.confidence = 1.0;
            // Drop needs_review for this measure upon edit
            currentParsedSheet.issues = (currentParsedSheet.issues || []).filter(
              i => !(i.severity === "needs_review" && i.measure_index === m.index)
            );
            renderStep3();
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
            currentParsedSheet.issues = (currentParsedSheet.issues || []).filter(
              i => !(i.severity === "needs_review" && i.measure_index === m.index)
            );
            renderStep3();
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
          m.chords.push({ raw: "1", beat: 1.0, bbox: null, confidence: 1.0, alternatives: [] });
          currentParsedSheet.issues = (currentParsedSheet.issues || []).filter(
            i => !(i.severity === "needs_review" && i.measure_index === m.index)
          );
          renderStep3();
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
        const err = await resp.json().catch(() => ({}));
        if (resp.status === 422) {
          const msg = typeof err.detail === "string" ? err.detail : (err.detail && err.detail.message ? err.detail.message : JSON.stringify(err.detail));
          alert(`和弦输入错误：${msg}`);
          return false;
        }
        throw new Error(err.detail || `保存失败 (HTTP ${resp.status})`);
      }
      const data = await resp.json();
      if (data && data.parsed) {
        currentParsedSheet = data.parsed;
        renderStep3();
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

    hideRenderError();
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
        const msg = err.detail || `生成伴奏失败 (HTTP ${resp.status})`;
        showRenderError(msg);
        throw new Error(msg);
      }

      const result = await resp.json();
      displayRenderResult(result, key, diff);
      setStep(4);
    } catch (err) {
      showRenderError(err.message);
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

  // Auto-load sheet from URL parameter (e.g. ?sheet_id=xxx)
  const urlParams = new URLSearchParams(window.location.search);
  const initialSheetId = urlParams.get("sheet_id");
  if (initialSheetId) {
    currentSheetId = initialSheetId;
    fetch(`/api/sheets/${initialSheetId}`)
      .then((r) => r.json())
      .then((state) => {
        if (state.status === "ready" && state.parsed) {
          currentParsedSheet = state.parsed;
          renderStep3();
          setStep(3);
        } else {
          setStep(2);
          startPolling(initialSheetId);
        }
      })
      .catch((err) => console.error("Failed to load initial sheet:", err));
  }
});
