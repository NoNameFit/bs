import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import Counter

# 基础配置
st.set_page_config(page_title="车牌识别和结果管理管理系统", layout="wide")

DB_PATH = 'car_system_v2.db'
FONT_PATH = 'simhei.ttf' # 字体文件
AUTO_GRADIENTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3] # 自动寻优梯度
SAMPLE_COUNT = 5 # 投票采样次数

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    # 日志表
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, time TEXT, 
                  raw_image BLOB, result TEXT, conf REAL, note TEXT)''')
    # 分析表
    c.execute('''CREATE TABLE IF NOT EXISTS analysis 
                 (img_hash TEXT PRIMARY KEY, username TEXT, raw_image BLOB, 
                  result TEXT, is_modified TEXT DEFAULT '否')''')
    c.execute("INSERT OR IGNORE INTO users VALUES ('admin', 'admin', 'admin')")
    conn.commit()
    conn.close()

def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

init_db()

# 状态管理
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'current_conf' not in st.session_state:
    st.session_state.current_conf = 0.5
if 'last_uploaded_hash' not in st.session_state:
    st.session_state.last_uploaded_hash = None

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()

# 核心算法
@st.cache_resource
def load_models():
    yolo = YOLO('best.pt')
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    return yolo, ocr

def apply_clahe(img_np):
    """图像增强：自适应直方图均衡化"""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def draw_chinese_text(image, text, position, font_path, font_size=36, color=(0, 255, 0)):
    """在图片上绘制框"""
    cv2_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def perform_inference_logic(img_np, conf):
    """YOLO检测 + PaddleOCR识别 + 多次采样投票"""
    yolo_model, ocr_engine = load_models()
    results = yolo_model.predict(source=img_np, conf=conf, verbose=False)
    found_texts = []
    
    # 模拟采样投票流程
    for _ in range(SAMPLE_COUNT):
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                crop = img_np[max(0,y1):min(img_np.shape[0],y2), max(0,x1):min(img_np.shape[1],x2)]
                if crop.size > 0:
                    res = ocr_engine.ocr(crop, cls=True)
                    if res and res[0]:
                        found_texts.append("".join([line[1][0] for line in res[0]]))
    
    annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    voted_res = []
    if found_texts:
        voted_text = Counter(found_texts).most_common(1)[0][0]
        voted_res = [voted_text]
        # 框出车牌位置
        if results and len(results[0].boxes) > 0:
            b = results[0].boxes.xyxy.cpu().numpy()[0]
            cv2.rectangle(annotated_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
            
    return annotated_bgr, voted_res

# UI 界面
if not st.session_state.logged_in:
    # 登录/注册页面
    st.title("🚗 车牌识别和结果管理管理系统")
    tab1, tab2 = st.tabs(["🔒 登录", "📝 注册"])
    with tab1:
        u = st.text_input("用户名")
        p = st.text_input("密码", type='password')
        if st.button("登录系统", use_container_width=True):
            conn = sqlite3.connect(DB_PATH); res = conn.execute("SELECT role FROM users WHERE username=? AND password=?", (u, p)).fetchone(); conn.close()
            if res: st.session_state.logged_in = True; st.session_state.user = u; st.session_state.role = res[0]; st.rerun()
            else: st.error("账号不存在或密码错误")
    with tab2:
        new_u = st.text_input("输入新用户名")
        new_p = st.text_input("输入新密码", type='password')
        if st.button("注册账号"):
            try:
                conn = sqlite3.connect(DB_PATH); conn.execute("INSERT INTO users VALUES (?, ?, 'user')", (new_u, new_p)); conn.commit(); conn.close()
                st.success("注册成功！请前往登录页登录。")
            except: st.error("账号已存在")

else:
    # 管理员界面
    if st.session_state.role == 'admin':
        st.sidebar.title("🛠️ 管理员面板")
        choice = st.sidebar.radio("导航栏", ["用户管理", "数据审查", "个人中心"])
        if choice == "用户管理":
            st.header("👥 用户账号管理")
            conn = sqlite3.connect(DB_PATH)
            # 展示用户列表
            users_df = pd.read_sql("SELECT username AS 账号, role AS 角色 FROM users", conn)
            st.dataframe(users_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("永久注销账号")
            del_u = st.text_input("输入要彻底删除的账号名称", help="警告：此操作将同步清空该用户的所有数据")
            
            if st.button("确认永久删除", type="primary"):
                if del_u == 'admin' or del_u == st.session_state.user:
                    st.error("无法删除管理员账号！")
                else:
                    try:
                        # 开启事务处理，确保三表同步删除的原子性
                        c = conn.cursor()
                        # 1. 删除用户表记录
                        c.execute("DELETE FROM users WHERE username = ?", (del_u,))
                        # 2. 删除日志表关联数据
                        c.execute("DELETE FROM logs WHERE username = ?", (del_u,))
                        # 3. 删除分析表关联数据
                        c.execute("DELETE FROM analysis WHERE username = ?", (del_u,))
                        
                        conn.commit()
                        st.success(f"用户 [{del_u}] 及其所有关联业务数据已从系统中完全抹除。")
                        st.rerun()
                    except Exception as e:
                        conn.rollback()
                        st.error(f"删除失败：{str(e)}")
            conn.close()
        
        elif choice == "数据审查":
            st.header("🔍 全局数据审查中心")
            conn = sqlite3.connect(DB_PATH)
            
            # 搜索组件
            search_user = st.text_input("请输入要查询的用户账号", placeholder="输入用户名即可")
            
            if search_user:
                st.subheader(f"账户 [{search_user}] 的识别日志")
                # 展示日志的时间、哈希、结果、置信度、备注
                # 通过SQL关联analysis表获取，或直接展示logs关键信息
                query = """
                    SELECT l.time, a.img_hash, l.result, l.conf, l.note 
                    FROM logs l
                    LEFT JOIN analysis a ON l.raw_image = a.raw_image
                    WHERE l.username = ?
                    ORDER BY l.time DESC
                """
                audit_df = pd.read_sql(query, con=conn, params=(search_user,))
                
                if not audit_df.empty:
                    # 重命名列名以符合预览要求
                    audit_df.columns = ['识别时间', '车牌哈希值', '识别结果', '置信度', '备注']
                    st.dataframe(audit_df, use_container_width=True)
                else:
                    st.warning(f"未找到用户 [{search_user}] 的相关识别记录。")
            else:
                st.info("请在上方输入框输入用户账号以调取审计日志。")
            conn.close()
        
        elif choice == "个人中心":
            new_p = st.text_input("输入新密码", type='password')
            if st.button("完成修改"):
                conn = sqlite3.connect(DB_PATH); conn.execute("UPDATE users SET password=? WHERE username=?", (new_p, st.session_state.user)); conn.commit(); conn.close(); st.success("完成")
            if st.sidebar.button("退出登录"): logout()

    # 普通用户界面
    else:
        st.sidebar.title(f"👋欢迎！ {st.session_state.user}")
        nav = st.sidebar.radio("系统导航栏", ["车牌识别", "数据分析", "个人中心"])

        if nav == "车牌识别":
            st.header("📸 智能识别板块")
            with st.sidebar:
                st.markdown("---")
                enable_enhancement = st.checkbox("开启画质增强 (CLAHE)", value=True)
                manual_conf = st.slider("手动调节置信度阈值", 0.05, 1.0, value=st.session_state.current_conf, step=0.05)
                if manual_conf != st.session_state.current_conf:
                    st.session_state.current_conf = manual_conf

            uploaded_file = st.file_uploader("📂 上传车辆图像", type=['jpg','png','jpeg'])
            if uploaded_file:
                img_bytes = uploaded_file.getvalue()
                img_hash = get_image_hash(img_bytes)
                img_pil = Image.open(uploaded_file)
                img_np = np.array(img_pil)

                # 梯度分析自动寻优
                if st.session_state.last_uploaded_hash != img_hash:
                    with st.spinner('执行梯度置信度分析中...'):
                        for g in AUTO_GRADIENTS:
                            _, res = perform_inference_logic(img_np, g)
                            if res:
                                st.session_state.current_conf = g
                                st.session_state.last_uploaded_hash = img_hash
                                st.rerun()
                                break
                    st.session_state.last_uploaded_hash = img_hash

                # 检查分析表是否被人工修改
                conn = sqlite3.connect(DB_PATH)
                existing = conn.execute("SELECT result, is_modified FROM analysis WHERE img_hash=?", (img_hash,)).fetchone()
                
                # 判定识别流程
                final_res, final_conf, note = "", 0.0, ""
                process_img = apply_clahe(img_np) if enable_enhancement else img_np
                
                if existing and existing[1] == '是':
                    final_res, final_conf, note = existing[0], 1.0, "基于分析表人工修改结果"
                    # 绘制显示图
                    final_img_bgr, _ = perform_inference_logic(process_img, 0.3) 
                else:
                    final_img_bgr, res_list = perform_inference_logic(process_img, st.session_state.current_conf)
                    if res_list:
                        final_res, final_conf = res_list[0], st.session_state.current_conf
                    else:
                        final_res, final_conf, note = "未识别出有效车牌号", st.session_state.current_conf, "未能识别，请调节阈值重试"

                # 页面展示
                col1, col2 = st.columns(2)
                with col1: st.image(img_pil, caption="原始照片", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB), caption="分析结果 (框选定位)", use_container_width=True)
                    if final_res != "未识别出有效车牌号":
                        st.success(f"判定车牌：{final_res}")
                        st.info(f"说明：采用梯度分析锁定最佳阈值，置信度为 {final_conf} ")
                    else:
                        st.error(note)

                # 数据入库
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("INSERT INTO logs (username, time, raw_image, result, conf, note) VALUES (?,?,?,?,?,?)",
                             (st.session_state.user, now, img_bytes, final_res, final_conf, note))
                conn.execute("INSERT OR REPLACE INTO analysis (img_hash, username, raw_image, result) VALUES (?,?,?,?)",
                             (img_hash, st.session_state.user, img_bytes, final_res))
                conn.commit(); conn.close()

        elif nav == "数据分析":
            st.header("📊 数据分析修正与日志导出")
            conn = sqlite3.connect(DB_PATH)
            
            st.subheader("🔍 分析表基准数据")
            # 从数据库读取原始照片、结果和修改状态
            query = "SELECT img_hash, raw_image, result, is_modified FROM analysis WHERE username=?"
            c = conn.cursor()
            c.execute(query, (st.session_state.user,))
            rows = c.fetchall()

            if not rows:
                st.info("暂无分析数据，请先前往识别板块上传图片。")
            else:
                # 建立表头
                head1, head2, head3, head4 = st.columns([2, 2, 1, 1.5])
                with head1: st.markdown("**原始照片**")
                with head2: st.markdown("**识别结果**")
                with head3: st.markdown("**状态**")
                with head4: st.markdown("**操作**")
                st.divider()

                for row in rows:
                    h_val, img_data, r_val, m_val = row
                    img_pil = Image.open(io.BytesIO(img_data))
                    
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 1.5])
                    
                    with col1:
                        with st.expander("查看大图"):
                            st.image(img_pil, use_container_width='stretch')
                        st.image(img_pil, width=150)

                    with col2:
                        new_input = st.text_input(f"结果_{h_val}", value=r_val, label_visibility="collapsed", key=f"input_{h_val}")

                    with col3:
                        status_color = "green" if m_val == "是" else "gray"
                        st.markdown(f":{status_color}[{m_val}]")

                    with col4:
                        # 创建子列以并排显示按钮
                        btn_col1, btn_col2 = st.columns(2)
                        
                        # 1. 保存修改按钮
                        if btn_col1.button("保存修改", key=f"save_{h_val}"):
                            if new_input != r_val:
                                conn.execute("UPDATE analysis SET result=?, is_modified='是' WHERE img_hash=?", (new_input, h_val))
                                conn.execute("""
                                    UPDATE logs 
                                    SET result=?, note='已进行人工修改' 
                                    WHERE username=? AND raw_image=(SELECT raw_image FROM analysis WHERE img_hash=?)
                                """, (new_input, st.session_state.user, h_val))
                                conn.commit()
                                st.success("✅ 已同步修正！")
                                st.rerun()
                            else:
                                st.warning("未变动")

                        # 2. 删除记录按钮
                        if btn_col2.button("删除记录", key=f"del_{h_val}"):
                            # 执行删除逻辑
                            # A. 更新日志表备注
                            conn.execute("""
                                UPDATE logs 
                                SET note = '已从分析表删除' 
                                WHERE username=? AND raw_image=(SELECT raw_image FROM analysis WHERE img_hash=?)
                            """, (st.session_state.user, h_val))
                            
                            # B. 从分析表中物理删除
                            conn.execute("DELETE FROM analysis WHERE img_hash=?", (h_val,))
                            
                            conn.commit()
                            st.error("已移除记录，日志已更新备注。")
                            st.rerun()
                            
                st.divider()

            # 日志表导出
            st.subheader("📜 历史识别日志导出")
            # 重新查询包含图片二进制数据的日志
            logs_query = "SELECT username, time, raw_image, result, conf, note FROM logs WHERE username=?"
            c = conn.cursor()
            c.execute(logs_query, (st.session_state.user,))
            full_logs = c.fetchall()

            if full_logs:
                if st.button("📥 导出 Excel 报表"):
                    output = io.BytesIO()
                    # 使用 xlsxwriter 作为引擎
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        workbook = writer.book
                        worksheet = workbook.add_worksheet('识别记录')
                        
                        # 设置表头
                        headers = ['用户名', '时间', '车牌图片', '识别结果', '置信度', '备注']
                        for col_num, header in enumerate(headers):
                            worksheet.write(0, col_num, header)
                        
                        # 设置行高以便图片能看清
                        for row_num, data in enumerate(full_logs, start=1):
                            u_name, t_time, img_blob, res, cnfd, nt = data
                            
                            worksheet.set_row(row_num, 60) # 设置行高为 60
                            worksheet.write(row_num, 0, u_name)
                            worksheet.write(row_num, 1, t_time)
                            
                            # 嵌入图片逻辑
                            if img_blob:
                                image_data = io.BytesIO(img_blob)
                                # 插入图片调整缩放比例以适应单元格
                                worksheet.insert_image(row_num, 2, "plate.png", 
                                                     {'image_data': image_data, 
                                                      'x_scale': 0.15, 'y_scale': 0.15,
                                                      'x_offset': 5, 'y_offset': 5})
                            
                            worksheet.write(row_num, 3, res)
                            worksheet.write(row_num, 4, cnfd)
                            worksheet.write(row_num, 5, nt)
                        
                        # 设置列宽
                        worksheet.set_column('A:B', 20)
                        worksheet.set_column('C:C', 15)
                        worksheet.set_column('D:F', 15)
                    
                    st.download_button(
                        label="下载文件",
                        data=output.getvalue(),
                        file_name=f"{st.session_state.user}_车牌识别日志报表.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            conn.close()

        elif nav == "个人中心":
            st.header("👤 信息修改")
            new_u = st.text_input("输入新账号", value=st.session_state.user)
            new_p = st.text_input("输入新密码", type='password')
            if st.button("确认修改并保存"):
                conn = sqlite3.connect(DB_PATH)
                try:
                    conn.execute("UPDATE users SET username=?, password=? WHERE username=?", (new_u, new_p, st.session_state.user))
                    conn.execute("UPDATE logs SET username=? WHERE username=?", (new_u, st.session_state.user))
                    conn.execute("UPDATE analysis SET username=? WHERE username=?", (new_u, st.session_state.user))
                    conn.commit(); st.session_state.user = new_u; st.success("信息已同步迁移")
                except: st.error("该账号名已被占用")
                conn.close()            
            if st.sidebar.button("退出登录"): logout()